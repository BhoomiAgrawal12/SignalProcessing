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

from ..common.models import ModulationHypothesis
from ..demod.filters import rrc_taps
from ..demod.receiver import _coarse_cfo, _timing_recover
from .cumulants import classify_cumulants
from .cvnet import classify_cvnet

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


def classify_modulation(x: np.ndarray, params, config) -> ModulationHypothesis:
    """x: channelised baseband signal; params: SignalParameters from S4."""
    constraints = []
    engines = {}
    amp = np.abs(np.asarray(x[: 1 << 18]))
    env_cv = float(amp.std() / (amp.mean() + 1e-12))

    # ---- hard physical constraints -------------------------------------
    # FSK first: a slow FSK's tone dwell also produces bursty lag
    # correlation that can mimic an OFDM cyclic prefix.
    if params.fsk_tone_count:
        label = f"{params.fsk_tone_count}FSK"
        conf = params.confidences.get("fsk", {}).get("value", 0.5)
        constraints.append(f"instantaneous-frequency histogram shows "
                           f"{params.fsk_tone_count} tones")
        return ModulationHypothesis(
            prediction=label, confidence=round(min(1.0, conf), 3),
            alternatives=[[label, round(min(1.0, conf), 3)]],
            engine_predictions={"fsk_histogram": [label, round(min(1.0, conf), 3)]},
            constraints_applied=constraints,
            in_distribution=label in ("2FSK", "4FSK"))

    sps = params.samples_per_symbol or 8.0

    # ---- engine A: cumulants -------------------------------------------
    syms = _prepare_symbols(x, sps)
    cum = None
    if syms is not None:
        cum = classify_cumulants(syms, params.snr_db)
        engines["cumulant"] = cum

    # ---- engine B: CVNet-RF --------------------------------------------
    cv = None
    if config.cvnet_enabled:
        cv = classify_cvnet(x, config.cvnet_checkpoint, config.cvnet_variant,
                            config.device, config.frame_size, config.max_frames)
        if cv and "probabilities" in cv:
            engines["cvnet"] = cv

    # ---- OFDM: only when no single-carrier structure was found ----------
    # (slow FSK tone dwell and single-carrier pulse autocorrelation both
    # mimic a cyclic prefix, so CP correlation alone is not sufficient)
    cum_top = max(cum["probabilities"].values()) if cum else 0.0
    # a strong single-carrier symbol-rate line is decisive AGAINST OFDM
    # (frame periodicity in single-carrier data can mimic cyclic-prefix
    # recurrence, but OFDM has no |x|^2 symbol-rate tone)
    sr_conf = params.confidences.get("symbol_rate", {}).get("value", 0)
    if params.ofdm_detected and env_cv > 0.35 and cum_top < 0.75 and \
            sr_conf < 0.5 and \
            (params.confidences.get("ofdm", {}).get("value", 0) > 0.6):
        return ModulationHypothesis(
            prediction="OFDM", confidence=params.confidences["ofdm"]["value"],
            alternatives=[["OFDM", params.confidences["ofdm"]["value"]]],
            engine_predictions={"cumulant": ["no clear constellation",
                                             round(cum_top, 3)]} if cum else {},
            constraints_applied=["cyclic-prefix autocorrelation",
                                 f"envelope cv {env_cv:.2f} consistent with OFDM",
                                 "no single-carrier cumulant structure"],
            in_distribution=False)

    # ---- fusion ---------------------------------------------------------
    classes = list(config.classes)
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
        s = sum(mapped.values())
        if s > 0.05:
            for c in mapped:
                fused[c] += w_cv * mapped[c] / s
            total_w += w_cv
    # constant envelope excludes QAM outright
    if env_cv < 0.15:
        for c in list(fused):
            if c.endswith("QAM"):
                fused[c] *= 0.05
        constraints.append(f"envelope cv {env_cv:.2f}: QAM excluded "
                           "(constant-envelope signal)")
    if total_w == 0:
        return ModulationHypothesis(prediction="UNKNOWN", confidence=0.0,
                                    constraints_applied=constraints,
                                    in_distribution=False)
    for c in fused:
        fused[c] /= total_w
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    top, top_p = ranked[0]

    engine_preds = {}
    if cum:
        cb = max(cum["probabilities"].items(), key=lambda kv: kv[1])
        engine_preds["cumulant"] = [cb[0], round(cb[1], 3)]
    if cv and "probabilities" in cv:
        vb = max(cv["probabilities"].items(), key=lambda kv: kv[1])
        engine_preds["cvnet"] = [vb[0], round(vb[1], 3)]
    agreement = True
    if len(engine_preds) == 2:
        a = engine_preds["cumulant"][0]
        b = CVNET_MAP.get(engine_preds["cvnet"][0], engine_preds["cvnet"][0])
        agreement = (a == b)

    hyp = ModulationHypothesis(
        prediction=top if top_p > 0.25 else "UNKNOWN",
        confidence=round(top_p if top_p > 0.25 else 0.0, 3),
        alternatives=[[c, round(p, 3)] for c, p in ranked[:5] if p > 0.01],
        engine_predictions=engine_preds,
        classifier_agreement=agreement,
        constraints_applied=constraints,
        in_distribution=(ood_mass < 0.5))
    if cum:
        hyp.engine_predictions["cumulant_features"] = dict(
            zip(cum["feature_names"], cum["features"]))
    return hyp
