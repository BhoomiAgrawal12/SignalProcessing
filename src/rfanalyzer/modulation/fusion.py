"""Stage S5 driver: physical pre-checks -> two engines -> fusion voter.

Order of authority (report S5 'Fusion'):
1. Hard physical constraints from S4 (FSK tone histogram, OFDM detection,
   constant-envelope test) can exclude classes outright.
2. Cumulant engine (explainable, no training data).
3. CVNet-RF (advisory; its 24 RadioML classes are mapped onto our class
   list and out-of-distribution mass is reported, not hidden).
Disagreement between engines is surfaced, never averaged away.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

from ..common.models import ModulationHypothesis, TrialRank
from ..demod.constellations import MOD_FAMILY
from ..demod.filters import rrc_taps
from ..demod.receiver import (_EVM_GATES, _coarse_cfo, _timing_recover,
                              demodulate)
from .cumulants import classify_cumulants
from .cvnet import classify_cvnet

_LINEAR_FAMILIES = ("psk", "oqpsk", "qam", "apsk", "ask")
# Continuous-phase families.  These used to leave classify_modulation
# through an early return, before any receiver ran (report §5.4): a
# GMSK capture at 10 dB was therefore accepted as OQPSK at 5.2% BER
# because the OQPSK receiver locked and nothing ever asked the GMSK one,
# and a 2FSK capture at 10 dB was reported UNKNOWN while its own
# demodulator was producing a bit-perfect stream.  They are trialled
# alongside the linear classes now.
_CPM_FAMILIES = ("fsk", "gmsk")
_TRIALABLE = _LINEAR_FAMILIES + _CPM_FAMILIES
# family representatives tried when the fused shortlist itself fails to
# lock: smeared clouds most often steal probability from these sparse
# tables, so they must always get a hearing
_TRIAL_FALLBACK = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM",
                   "4ASK", "16APSK", "2FSK", "4FSK", "GMSK"]
# acceptance bar for a receiver trial: a candidate at or above this
# locked well enough for its claim to stand on its own evidence
_TRIAL_ACCEPT = 2.0
# below this a lock is not evidence at all
_TRIAL_FLOOR = 0.8


def _occupancy(symbols, candidate) -> float:
    """Normalised entropy of the constellation hit histogram.

    Sparse tables embed in dense ones up to gain (QPSK corners ARE four
    16QAM points), so nearest-point EVM alone cannot reject the denser
    table; a true signal exercises all of its points, an embedded subset
    leaves most of them empty."""
    from ..demod.constellations import CONSTELLATIONS
    table, _k = CONSTELLATIONS[candidate]
    t = np.asarray(table, dtype=np.complex128)
    t = t / (np.sqrt((np.abs(t) ** 2).mean()) + 1e-12)
    sset = np.asarray(symbols[:8192], dtype=np.complex128)
    sset = sset / (np.sqrt((np.abs(sset) ** 2).mean()) + 1e-12)
    idx = np.argmin(np.abs(sset[:, None] - t[None, :]), axis=1)
    hist = np.bincount(idx, minlength=len(t)).astype(np.float64)
    pr = hist / hist.sum()
    pr = pr[pr > 0]
    h = float(-(pr * np.log2(pr)).sum())
    return h / max(np.log2(len(t)), 1.0)


def _grid_structure(symbols, candidate) -> float:
    """Decision-directed residual-bias structure score in [0, 1].

    A true lock leaves residuals CENTRED on each constellation point
    (per-point mean near zero); a structureless cloud quantised onto the
    same table leaves residuals biased along the cloud's local density
    gradient inside every decision region. Unlike an off-grid-rotation
    baseline this cannot be fooled by dense rings whose angular pitch
    divides the rotation angle."""
    from ..demod.constellations import CONSTELLATIONS
    table, _k = CONSTELLATIONS[candidate]
    t = np.asarray(table, dtype=np.complex128)
    t = t / (np.sqrt((np.abs(t) ** 2).mean()) + 1e-12)
    sset = np.asarray(symbols[:4096], dtype=np.complex128)
    sset = sset / (np.sqrt((np.abs(sset) ** 2).mean()) + 1e-12)
    idx = np.argmin(np.abs(sset[:, None] - t[None, :]), axis=1)
    resid = sset - t[idx]
    # project out the common linear error (residual gain, rotation, DC):
    # those are receiver imperfections that scale with the point, not
    # evidence of a cloud; a cloud's density-gradient bias is nonlinear
    # in position and survives this projection
    A = np.vstack([t[idx], np.ones_like(idx, dtype=np.complex128)]).T
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    resid = resid - A @ coef
    rms = float(np.sqrt((np.abs(resid) ** 2).mean())) + 1e-12
    biases = []
    for pt in np.unique(idx):
        r = resid[idx == pt]
        if len(r) >= 6:
            # noise-debiased: a centred Gaussian still shows |mean| of
            # about rms/sqrt(n), which must not count as cloud bias
            raw = float(np.abs(r.mean()))
            floor = float(np.sqrt((np.abs(r) ** 2).mean())) / np.sqrt(len(r))
            biases.append(max(0.0, raw - floor))
    if not biases:
        return 0.0
    bias_ratio = float(np.median(biases)) / rms
    return max(0.0, 1.0 - bias_ratio)


def _cpm_occupancy(res, order: int) -> float:
    """Normalised entropy of the FSK tone-usage histogram.

    The CPM counterpart of :func:`_occupancy`: a true 4FSK signal
    exercises all four tones, while a 2FSK signal scored against the
    4FSK hypothesis leaves two of them empty.  Without it a sparse tone
    set embeds in a denser one exactly the way a sparse constellation
    embeds in a denser grid.
    """
    if res.symbols is None or not len(res.symbols):
        return 0.0
    # _demod_fsk publishes symbols as (sampled tone frequency)/(spacing),
    # so the tone index is the nearest integer offset from the centre
    z = np.real(np.asarray(res.symbols, dtype=np.complex128))
    if order <= 2:
        idx = (z > 0).astype(int)
    else:
        idx = np.clip(np.round(z + (order - 1) / 2.0), 0,
                      order - 1).astype(int)
    hist = np.bincount(idx, minlength=order).astype(np.float64)
    if hist.sum() <= 0:
        return 0.0
    pr = hist / hist.sum()
    pr = pr[pr > 0]
    h = float(-(pr * np.log2(pr)).sum())
    return h / max(np.log2(order), 1.0)


def _trial_one(x, candidate, sps, demod_cfg):
    """Trial-demodulate one candidate with the real S6 receiver and score
    how well it locked, on that family's own terms.

    Linear families are scored against their calibrated EVM gate: raw
    nearest-point EVM always improves with table density, so it is only
    comparable across candidates after normalising by the gate, and the
    occupancy factor then rejects the subset embeddings EVM cannot see.

    Continuous-phase families have no EVM at all; their equivalent is how
    tightly the samples sit on the tone set (M-FSK) or how far the
    frequency discriminator swings (GMSK).  Both are normalised by their
    own FAILED threshold so the resulting score means the same thing as
    the linear one - "how many times better than the minimum acceptable
    lock" - which is what makes a CPM candidate comparable with a linear
    one at all.  Without that comparison GMSK at 10 dB is reported as
    OQPSK and 2FSK at 10 dB as UNKNOWN (report §5.4).

    Returns a dict, or None if the receiver raised.
    """
    try:
        res = demodulate(x, candidate, sps, demod_cfg)
    except Exception:
        return None
    family = MOD_FAMILY.get(candidate)
    base = {"status": res.demodulation_status,
            "sps": round(float(sps), 3),
            "family": family or "",
            "score": None}
    locked = res.timing_locked and res.carrier_locked

    if family in _CPM_FAMILIES:
        metrics = {}
        # A continuous-phase signal has a constant envelope, and nothing
        # else in the CPM trial can substitute for that check: the
        # frequency discriminator swings hard on ANY signal, so the GMSK
        # margin and the M-FSK tone-fit both look respectable when fed a
        # 16QAM burst.  Measured on this project's own corpus, RRC-shaped
        # linear modulations sit at an envelope cv of 0.27 and above
        # while FSK and GMSK stay at 0.13 and below, so the gate is
        # decisive with a wide margin either side.
        amp = np.abs(np.asarray(x[: 1 << 16]))
        env_cv = float(amp.std() / (amp.mean() + 1e-12))
        cm = float(np.clip((0.20 - env_cv) / 0.07, 0.0, 1.0))
        metrics["envelope_cv"] = round(env_cv, 4)
        metrics["constant_envelope_factor"] = round(cm, 3)
        if cm <= 0.0:
            base.update({"occupancy": 0.0, "metrics": metrics,
                         "evm": None, "gate": None,
                         "reason": (f"envelope cv {env_cv:.2f}: not a "
                                    "constant-envelope signal, so no "
                                    "continuous-phase hypothesis applies")})
            return base
        if family == "gmsk":
            # ``discriminator_margin`` is the measured peak frequency
            # deviation divided by the deviation h=0.5 predicts, so a
            # true MSK-family signal scores 1.0 and the statistic is a
            # MODULATION-INDEX TEST, not a "bigger is better" quality
            # figure.  Treating it as the latter is what let GMSK win on
            # 4FSK, 16QAM and 16APSK captures: their discriminators swing
            # three to four times harder than h=0.5 allows, which the old
            # scoring read as an excellent lock.
            margin_raw = res.lock_metrics.get("discriminator_margin")
            metrics["discriminator_margin"] = margin_raw
            metrics["level_spread"] = res.lock_metrics.get("level_spread")
            if margin_raw:
                dev = (float(margin_raw) - 1.0) / 0.5
                quality = 3.0 * float(np.exp(-dev * dev))
            else:
                quality = 0.0
            # h=0.5 continuous phase leaves an x^2 line PAIR at
            # 2fc +- Rs/2; without it this is not MSK-family, whatever
            # the discriminator says
            lq = res.lock_metrics.get("squaring_line_quality")
            metrics["squaring_line_quality"] = lq
            if not lq or float(lq) < 20.0:
                quality *= 0.35
            # h-index cross-check.  The x^2 line PAIR measures h*Rs, so
            # its spacing alone cannot separate h=0.5 at rate Rs from
            # h=1 at rate Rs/2 - and the GMSK receiver will happily
            # "explain" a rectangular h=1 FSK signal by assuming half its
            # symbol rate, which lands the discriminator margin right on
            # the 1.0 that the modulation-index test rewards.  Rs has to
            # come from somewhere independent, and it does: the trial is
            # run at the rate S4 measured from the tone-transition line.
            # h=1 gives a squaring rate of 2*Rs, h=0.5 gives Rs.
            rs2 = res.lock_metrics.get("squaring_rs_norm")
            metrics["squaring_rs_norm"] = rs2
            rs_expected = (1.0 / sps) if sps else None
            if rs2 and rs_expected and \
                    abs(rs2 - 2.0 * rs_expected) < 0.3 * rs2:
                metrics["h_index"] = (
                    "squaring rate is twice the measured symbol rate: "
                    "h=1 keying, not MSK-family")
                quality *= 0.2
            # Symbol-clock test, from the receiver's own two independent
            # rate measurements rather than from S4 (which is exactly the
            # estimate that can be wrong here).  h=0.5 keying makes the
            # squaring-line spacing and the tone-transition line the same
            # number; an analog FM carrier has a fine squaring pair and no
            # transition line to agree with it.
            ratio = res.lock_metrics.get("squaring_vs_transition")
            metrics["squaring_vs_transition"] = ratio
            if ratio is None:
                metrics["symbol_clock"] = ("no tone-transition line: no "
                                           "symbol clock is observable")
                quality *= 0.15
            elif abs(ratio - 1.0) > 0.25:
                metrics["symbol_clock"] = (
                    f"the squaring rate is {ratio:.2f} times the "
                    "tone-transition rate; h=0.5 keying makes them equal")
                quality *= 0.15
            occ = 1.0          # binary CPM: both signs are always used
        else:
            ratio = res.lock_metrics.get("tone_fit_ratio")
            metrics["tone_fit_ratio"] = ratio
            # _demod_fsk gates at a mean tone-fit cost of a quarter of
            # the tone spacing
            quality = (0.25 / float(ratio)) if ratio and ratio > 0 else 0.0
            occ = _cpm_occupancy(res, int(candidate[0]))
        base.update({"occupancy": round(float(occ), 3), "metrics": metrics,
                     "evm": None, "gate": None})
        if not locked or res.hard_bits is None or len(res.hard_bits) < 256:
            base["reason"] = ("receiver did not lock"
                              if not locked else "too few bits recovered")
            return base
        bonus = {"GOOD": 1.0, "DEGRADED": 0.5}.get(
            res.demodulation_status, 0.0)
        score = (bonus + min(3.0, quality)) * occ * cm
        base["score"] = round(float(score), 3)
        return base

    evm = res.evm_percent
    gate = _EVM_GATES.get(candidate)
    base["evm"] = evm
    base["gate"] = None if gate is None else gate[0]
    if (evm is None or gate is None or not locked or
            res.symbols is None or len(res.symbols) < 256):
        base["reason"] = ("receiver did not lock" if not locked else
                          "no EVM available" if evm is None else
                          "too few symbols")
        return base
    base["gate_ratio"] = round(float(evm) / gate[0], 3)
    margin = gate[0] / max(float(evm), 0.1)
    bonus = {"GOOD": 1.0, "DEGRADED": 0.5}.get(res.demodulation_status, 0.0)
    occ = _occupancy(res.symbols, candidate)
    base["occupancy"] = round(occ, 3)
    # off-grid-rotation baseline: a Gaussian blob sits at blob-level
    # nearest-point EVM on any dense grid (with full occupancy), so the
    # measured EVM only counts as evidence if the residuals around each
    # point are centred rather than biased by a density gradient
    structure = _grid_structure(res.symbols, candidate)
    base["structure"] = round(structure, 3)
    if structure < 0.5:
        base["reason"] = ("residuals are biased inside every decision "
                          "region: a cloud quantised onto this table, "
                          "not a lock")
        return base
    score = (bonus + min(3.0, margin)) * occ * min(1.0, structure / 0.8)
    base["score"] = round(float(score), 3)
    return base


# CVNet (RadioML) label -> our label; None = out of our supported set
CVNET_MAP = {
    "BPSK": "BPSK", "QPSK": "QPSK", "8PSK": "8PSK",
    "16QAM": "16QAM", "64QAM": "64QAM", "OQPSK": "QPSK",
    "GMSK": "2FSK", "OOK": "OOK",
}


def _prepare_symbols(x: np.ndarray, sps: float) -> np.ndarray:
    """CFO-correct + matched filter + blind timing so the cumulant engine
    sees symbol-spaced samples. Tries the plausible M-power CFO
    corrections and keeps the most structured result."""
    x = np.asarray(x, dtype=np.complex128)
    best_syms, best_metric = None, -1.0
    for order, limiter in ((2, True), (4, True), (4, False), (8, True)):
        cfo = _coarse_cfo(x, order, limiter=limiter)
        y = x * np.exp(-2j * np.pi * cfo * np.arange(len(x)))
        mf = sig.fftconvolve(y, rrc_taps(max(2, int(round(sps))), 10, 0.35),
                             mode="same")
        syms, tone, _ = _timing_recover(mf, sps, max_symbols=20000)
        if len(syms) < 64:
            continue
        syms = syms / (np.sqrt((np.abs(syms) ** 2).mean()) + 1e-12)
        # structure metric: strongest M-power concentration
        sn = syms / (np.abs(syms) + 1e-12)
        metric = max(np.abs((sn ** m).mean()) for m in (2, 4, 8))
        metric += float(np.abs((syms ** 4).mean()))
        if metric > best_metric:
            best_metric, best_syms = metric, syms
    return best_syms


def _sps_candidates(params, base_sps: float, limit: int = 3) -> list:
    """Alternate samples-per-symbol worth trialling.

    At low SNR the elected symbol-rate line is sometimes a data line, so
    the runners-up have to stay reachable.  The upper bound is 256 rather
    than the 64 the shipped code used: audio-rate telemetry runs at 40
    samples per symbol and higher, and a ceiling of 64 quietly removed
    those hypotheses from the search (report §10.5 item 2).
    """
    out = []
    for r in (getattr(params, "symbol_rate_candidates", []) or [])[1:8]:
        if not r:
            continue
        cand = 1.0 / max(float(r), 1e-9)
        if 1.9 <= cand <= 256 and abs(cand - base_sps) > 0.25 and \
                all(abs(cand - a) > 0.25 for a in out):
            out.append(cand)
        if len(out) >= limit:
            break
    return out


def classify_modulation(x: np.ndarray, params, config,
                        rank_all: bool = False) -> ModulationHypothesis:
    """x: channelised baseband signal; params: SignalParameters from S4.

    The decision rule is unchanged - a candidate is accepted when the
    real S6 receiver locks on it - but three things it used to hide are
    now published (report §8):

    * every candidate the classifier considered appears in
      ``trial_ranking``, including the ones that were never trialled,
      marked ``measured=False``.  "Tested and rejected" and "never
      tested" are different statements and the shipped output conflated
      them, which is why the IQ-02 misclassification was invisible: the
      true class sat fifth in the fused prior, only the top three were
      trialled, and 16QAM would have scored 3.88 against the accepted
      64QAM's failure had anyone asked it.
    * the fused-prior winner and the receiver-trial winner are both
      named, and a disagreement is flagged rather than silently resolved.
    * ``rank_all`` trials every trialable class.  It is not the default
      because a full sweep of the 18 linear classes costs about 4.1 s
      against a 6.3 s end-to-end runtime, with the four densest
      candidates accounting for 2.6 s of it.
    """
    constraints = []
    engines = {}
    amp = np.abs(np.asarray(x[: 1 << 18]))
    env_cv = float(amp.std() / (amp.mean() + 1e-12))
    classes = list(config.classes)
    rank_all = bool(rank_all or getattr(config, "rank_all", False))

    # ---- engine A: cumulants -------------------------------------------
    sps = params.samples_per_symbol or 8.0
    syms = _prepare_symbols(x, sps)
    cum = None
    if syms is not None:
        # Only pass the SNR when it is a measurement.  cumulants.py
        # widens its posterior below 15 dB, and the shipped S4 estimate
        # never reached 15 dB for QAM or APSK because the moment
        # estimator saturated there - so the classifier ran permanently
        # in its least confident mode for exactly the constellations that
        # need the sharpest one (report §3.3).
        snr_in = params.snr_db if getattr(params, "snr_state",
                                          "valid") == "valid" else None
        cum = classify_cumulants(syms, snr_in)
        engines["cumulant"] = cum

    # ---- engine B: CVNet-RF --------------------------------------------
    cv = None
    if config.cvnet_enabled:
        cv = classify_cvnet(x, config.cvnet_checkpoint, config.cvnet_variant,
                            config.device, config.frame_size, config.max_frames)
        if cv and "probabilities" in cv:
            engines["cvnet"] = cv

    # ---- physical evidence: prior shaping, not a verdict ----------------
    # The FSK tone histogram and the h=0.5 squaring signature used to
    # RETURN from here, before any receiver ran.  They are strong
    # evidence and they now say so by weighting the prior, but the
    # receiver still gets to decide (report §5.4).
    cpm_prior = {}
    if params.fsk_tone_count:
        from ..demod.receiver import _msk_squaring_lines
        rs2, _fc2, q2 = _msk_squaring_lines(np.asarray(x[: 1 << 17]))
        rs_t = params.symbol_rate_norm
        # h-index disambiguation: the x^2 line spacing measures h*Rs, so
        # Rs must come independently from the transition line.  h=1
        # rectangular FSK gives rs2 = 2*Rs; h=0.5 GMSK gives rs2 = Rs.
        is_h1_fsk = bool(rs_t and rs2 > 0 and
                         abs(rs2 - 2 * rs_t) < 0.3 * rs2)
        label = f"{params.fsk_tone_count}FSK"
        conf = float(params.confidences.get("fsk", {}).get("value", 0.5))
        if label in classes:
            cpm_prior[label] = min(1.0, conf)
        constraints.append(
            f"instantaneous-frequency histogram shows "
            f"{params.fsk_tone_count} tones")
        if q2 > 60 and not is_h1_fsk and "GMSK" in classes:
            cpm_prior["GMSK"] = max(cpm_prior.get("GMSK", 0.0),
                                    min(1.0, q2 / 120))
            constraints.append(
                "constant envelope with an x^2 line pair at 2fc +- Rs/2 "
                "(h=0.5 continuous-phase signature)")
        elif "GMSK" in classes:
            cpm_prior["GMSK"] = max(cpm_prior.get("GMSK", 0.0), 0.2)
    elif env_cv < 0.15:
        # constant envelope without a resolvable tone histogram: CPM is
        # still very much on the table
        for c in ("2FSK", "4FSK", "GMSK"):
            if c in classes:
                cpm_prior[c] = 0.25

    # ---- OFDM: only when no single-carrier structure was found ----------
    cum_top = max(cum["probabilities"].values()) if cum else 0.0
    sr_conf = params.confidences.get("symbol_rate", {}).get("value", 0)
    if params.ofdm_detected and env_cv > 0.35 and cum_top < 0.75 and \
            sr_conf < 0.5 and \
            (params.confidences.get("ofdm", {}).get("value", 0) > 0.6):
        return ModulationHypothesis(
            prediction="OFDM", confidence=params.confidences["ofdm"]["value"],
            alternatives=[["OFDM", params.confidences["ofdm"]["value"]]],
            engine_predictions={"cumulant": ["no clear constellation",
                                             round(cum_top, 3)]} if cum else {},
            constraints_applied=constraints + [
                "cyclic-prefix autocorrelation",
                f"envelope cv {env_cv:.2f} consistent with OFDM",
                "no single-carrier cumulant structure"],
            prior_winner="OFDM", trial_winner=None,
            in_distribution=False)

    # ---- fusion ---------------------------------------------------------
    fused = {c: 0.0 for c in classes}
    w_cum = config.fusion_weights.get("cumulant", 0.5)
    w_cv = config.fusion_weights.get("cvnet", 0.35)
    total_w = 0.0
    if cum:
        for c, p in cum["probabilities"].items():
            if c in fused:
                fused[c] += w_cum * p
        total_w += w_cum
    ood_mass = 0.0
    if cv and "probabilities" in cv:
        mapped = {c: 0.0 for c in classes}
        for label, p in cv["probabilities"].items():
            ours = CVNET_MAP.get(label)
            if ours and ours in mapped:
                mapped[ours] += p
            else:
                ood_mass += p
        s_cv = sum(mapped.values())
        if s_cv > 0.05:
            for c in mapped:
                fused[c] += w_cv * mapped[c] / s_cv
            total_w += w_cv
    if total_w > 0:
        for c in fused:
            fused[c] /= total_w
    # constant envelope excludes QAM outright
    if env_cv < 0.15:
        for c in list(fused):
            if c.endswith("QAM"):
                fused[c] *= 0.05
        constraints.append(f"envelope cv {env_cv:.2f}: QAM excluded "
                           "(constant-envelope signal)")
    # physical CPM evidence enters the prior at full weight
    for c, p in cpm_prior.items():
        fused[c] = max(fused.get(c, 0.0), float(p))
    if not any(v > 0 for v in fused.values()):
        return ModulationHypothesis(prediction="UNKNOWN", confidence=0.0,
                                    constraints_applied=constraints,
                                    in_distribution=False)
    total = sum(fused.values()) or 1.0
    fused = {c: v / total for c, v in fused.items()}
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    prior_winner = ranked[0][0]
    top, top_p = ranked[0]

    # ---- receiver-trial arbitration -------------------------------------
    from ..common.config import DemodConfig
    demod_cfg = DemodConfig()
    xs = np.asarray(x[: 1 << 16], dtype=np.complex128)
    trial_detail = {}
    trial_mode = "shortlist"

    def _sc(c):
        t = trial_detail.get(c)
        return -1.0 if not t or t.get("score") is None else t["score"]

    def _run(cands):
        for c in cands:
            if c in classes and c not in trial_detail and \
                    MOD_FAMILY.get(c) in _TRIALABLE:
                trial_detail[c] = _trial_one(xs, c, sps, demod_cfg)

    def _best():
        return max(trial_detail, key=_sc) if trial_detail else None

    trialable_ranked = [c for c, _p in ranked
                        if MOD_FAMILY.get(c) in _TRIALABLE]
    if rank_all:
        trial_mode = "all"
        _run(trialable_ranked)
        best = _best()
    else:
        _run(trialable_ranked[:3])
        best = _best()
        if best is None or _sc(best) < _TRIAL_ACCEPT:
            # nothing locked cleanly: give the family representatives a
            # hearing before concluding anything
            trial_mode = "escalated: family representatives"
            _run(_TRIAL_FALLBACK)
            best = _best()
        if best is None or _sc(best) < _TRIAL_ACCEPT:
            # representatives did not lock either: every remaining
            # trialable class gets a hearing in fused-probability order
            trial_mode = "escalated: all classes"
            for c in trialable_ranked:
                if c not in trial_detail:
                    _run([c])
                    if _sc(c) >= _TRIAL_ACCEPT:
                        break
            best = _best()
        if best is None or _sc(best) < _TRIAL_ACCEPT:
            # still nothing: revisit the runner-up symbol rates
            trial_mode = "escalated: alternate symbol rates"
            for s_alt in _sps_candidates(params, sps):
                for c in _TRIAL_FALLBACK[:7]:
                    if c not in classes or MOD_FAMILY.get(c) not in _TRIALABLE:
                        continue
                    t = _trial_one(xs, c, s_alt, demod_cfg)
                    if t and (t.get("score") or 0) > max(0.0, _sc(c)):
                        trial_detail[c] = t
                best = _best()
                if best is not None and _sc(best) >= _TRIAL_ACCEPT:
                    break

    # Subset-embedding sweep.  A sparse alphabet embeds in a denser one
    # of the same family - 16PSK's points ARE sixteen of 32PSK's, QPSK's
    # corners ARE four of 16QAM's - so a dense candidate can clear the
    # acceptance bar on a sparse signal and stop the search before the
    # true class is ever measured.  That is the IQ-02 failure exactly
    # (report §8.1: 16QAM sat fifth in the prior, only the top three were
    # trialled, and it would have scored 3.88 against the accepted
    # 64QAM's failure).  Occupancy detects the embedding, but only for a
    # candidate that was actually run, so whenever a candidate wins,
    # every SPARSER member of its family is measured too.
    if best is not None and MOD_FAMILY.get(best) in _LINEAR_FAMILIES:
        from ..demod.constellations import CONSTELLATIONS
        win_order = len(CONSTELLATIONS[best][0])
        sparser = [c for c in classes
                   if MOD_FAMILY.get(c) == MOD_FAMILY.get(best)
                   and c in CONSTELLATIONS
                   and len(CONSTELLATIONS[c][0]) < win_order
                   and c not in trial_detail]
        if sparser:
            _run(sparser)
            new_best = _best()
            if new_best is not None and _sc(new_best) > _sc(best):
                constraints.append(
                    f"{best} cleared the acceptance bar, but the sparser "
                    f"{new_best} of the same family scores higher: a "
                    "sparse alphabet embeds in a denser one, so the "
                    "denser hypothesis is only credible once the sparser "
                    "ones have been measured")
                best = new_best
            if trial_mode == "shortlist":
                trial_mode = "shortlist + subset sweep"

    trial_winner = best if best is not None and _sc(best) > 0 else None
    if best is not None and _sc(best) >= _TRIAL_FLOOR:
        if best != top:
            t_best = trial_detail[best]
            t_top = trial_detail.get(top)
            loser = ("was not trialled" if t_top is None else
                     "fails its own lock gate" if t_top.get("score") is None
                     else "locks worse relative to its gate")
            constraints.append(
                f"receiver trial overruled {top}: {best} locks "
                f"({t_best['status']}, score {t_best['score']}) while "
                f"{top} {loser}")
        # probability mass only over candidates that clear the acceptance
        # bar: sub-threshold locks stay visible in the ranking but must
        # not dilute the winner below the UNKNOWN cutoff by headcount
        scored = sorted((c for c in trial_detail if _sc(c) >= _TRIAL_FLOOR),
                        key=lambda c: (-_sc(c), -fused.get(c, 0.0)))
        exps_t = {c: float(np.exp(_sc(c))) for c in scored}
        z_t = sum(exps_t.values()) or 1.0
        trial_probs = {c: min(0.95, exps_t[c] / z_t) for c in scored}
        ranked = ([(c, trial_probs[c]) for c in scored] +
                  [(c, 0.0) for c, _p in ranked if c not in scored])
        top, top_p = ranked[0]
    else:
        constraints.append(
            "no candidate locks in the receiver trial: modulation claim "
            "withheld")
        top_p = min(top_p, 0.2)

    # ---- ranked trial table (report §8.3) --------------------------------
    table = []
    for c in classes:
        t = trial_detail.get(c)
        row = TrialRank(candidate=c, prior=round(float(fused.get(c, 0.0)), 4),
                        family=MOD_FAMILY.get(c, ""))
        if t is None:
            row.reason = ("not trialled: the search stopped once a "
                          "candidate cleared the acceptance bar"
                          if not rank_all else
                          "receiver raised on this candidate")
        else:
            row.measured = True
            row.score = t.get("score")
            row.evm_percent = t.get("evm")
            row.gate_good = t.get("gate")
            row.gate_ratio = t.get("gate_ratio")
            row.occupancy = t.get("occupancy")
            row.structure = t.get("structure")
            row.status = t.get("status", "FAILED")
            row.samples_per_symbol = t.get("sps")
            row.metrics = t.get("metrics", {}) or {}
            row.reason = t.get("reason", "")
        table.append(row)
    table.sort(key=lambda r: (r.score is None, -(r.score or 0.0), -r.prior))

    engine_preds = {}
    if cum:
        cb = max(cum["probabilities"].items(), key=lambda kv: kv[1])
        engine_preds["cumulant"] = [cb[0], round(cb[1], 3)]
    if cv and "probabilities" in cv:
        vb = max(cv["probabilities"].items(), key=lambda kv: kv[1])
        engine_preds["cvnet"] = [vb[0], round(vb[1], 3)]
    agreement = True
    if len(engine_preds) == 2:
        a_lab = engine_preds["cumulant"][0]
        b_lab = CVNET_MAP.get(engine_preds["cvnet"][0],
                              engine_preds["cvnet"][0])
        agreement = (a_lab == b_lab)

    disagree = bool(trial_winner and trial_winner != prior_winner)
    if disagree:
        constraints.append(
            f"fused prior favoured {prior_winner} but the receiver trial "
            f"winner is {trial_winner}: the trial is the stronger "
            "evidence, and the disagreement is reported rather than "
            "resolved silently")

    hyp = ModulationHypothesis(
        prediction=top if top_p > 0.25 else "UNKNOWN",
        confidence=round(top_p if top_p > 0.25 else 0.0, 3),
        alternatives=[[c, round(p, 3)] for c, p in ranked[:5] if p > 0.01],
        engine_predictions=engine_preds,
        classifier_agreement=agreement,
        constraints_applied=constraints,
        in_distribution=(ood_mass < 0.5),
        trial_ranking=table,
        prior_winner=prior_winner,
        trial_winner=trial_winner,
        trial_disagreement=disagree,
        trials_run=sum(1 for r in table if r.measured),
        trial_mode=trial_mode)
    if cum:
        hyp.engine_predictions["cumulant_features"] = dict(
            zip(cum["feature_names"], cum["features"]))
    if trial_detail:
        hyp.engine_predictions["receiver_trial"] = {
            c: t for c, t in trial_detail.items() if t is not None}
    return hyp
