"""Ground-truth validation matrix.

For each case a signal with full ground truth is generated and measured
on three things:

  1. blind classification (S5) - top-1, top-2 and the RANK OF TRUTH in
     the receiver-trial table;
  2. demodulation with the true modulation (S6) - status, EVM, BER;
  3. the fully blind chain (S0-S11) - frame detection, CRC, payload trust
     and payload ground-truth equality.

Verdicts per case:
  PASS      blind class top-1 correct AND BER < 2%
  DEGRADED  BER < 2% but classification missed top-1, or demod DEGRADED
  FAIL      unreliable bits (and honestly gated) or chain failure

What changed after the review (report §6, §11 item 11).  The shipped
harness ran ONE seed at ONE sample rate with only a fixed CFO and phase
offset, never enabled FEC, interleaving or scrambling, never varied
samples-per-symbol away from 8, and never touched a real recording - so
the 49-case result described a much narrower thing than "the matrix
passes".  It also scored BER by picking the best alignment, rotation and
conjugation over a 1,200-bit window, which is a biased estimator and
produced the impossible pairing of 8.2% EVM with 0.0 BER for 128APSK.
This version:

  * runs several seeds and reports the spread, not just a point value;
  * sweeps the impairments the factory already implements - phase noise,
    timing offset, clock ppm, IQ imbalance, multipath, DC offset - which
    were implemented and never exercised;
  * sweeps samples-per-symbol from 2.5 to 40, covering both the receiver's
    sub-3-sps path and audio-rate telemetry;
  * runs bit-layer cases with FEC, interleaving and scrambling enabled;
  * includes the AO-73 real recording with its published ground truth;
  * scores BER with a FIXED alignment chosen from a prefix and applied to
    the whole stream, and reports a Wilson interval on it.

Run:  .venv/bin/python scripts/validate_matrix.py [--quick] [--seeds 3]
      [--suite core|impairments|sps|bitlayer|real|all]
"""
import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.disable(logging.INFO)

from rfanalyzer.common.config import Config
from rfanalyzer.conditioning import condition
from rfanalyzer.detection import detect_signals
from rfanalyzer.channelization import channelize
from rfanalyzer.params import estimate_parameters
from rfanalyzer.modulation import classify_modulation
from rfanalyzer.demod import demodulate
from rfanalyzer.demod.constellations import (CONSTELLATIONS, MOD_FAMILY,
                                             slice_symbols, symmetry_order)
from rfanalyzer.ingestion import load_recording
from rfanalyzer.pipeline import RFAnalyzer
from rfanalyzer.synth.factory import WaveformFactory

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# modulation -> SNR levels to test (family-appropriate: denser needs more)
MATRIX = {
    "BPSK": [8, 15], "QPSK": [12, 20], "OQPSK": [15, 22],
    "8PSK": [18, 25], "16PSK": [25, 32], "32PSK": [30, 37],
    "OOK": [10, 15], "4ASK": [18, 25], "8ASK": [24, 30],
    "16QAM": [18, 25], "32QAM": [21, 28], "64QAM": [24, 30],
    "128QAM": [28, 34], "256QAM": [31, 37],
    "16APSK": [20, 27], "32APSK": [24, 30], "64APSK": [27, 33],
    "128APSK": [32, 38],
    "2FSK": [10, 15], "4FSK": [14, 20], "GMSK": [10, 15],
    "OFDM": [20, 25],
    "FM": [15], "AM-DSB-WC": [15], "AM-DSB-SC": [15],
    "AM-SSB-WC": [15], "AM-SSB-SC": [15],
}
ANALOG = {"FM", "AM-DSB-WC", "AM-DSB-SC", "AM-SSB-WC", "AM-SSB-SC"}

# Impairments the factory implements.  Every one of these was dead code
# as far as the harness was concerned (report §6).
IMPAIRMENTS = {
    "clean": {},
    "phase_noise": {"phase_noise_std": 0.004},
    "timing_offset": {"timing_frac": 0.37},
    "clock_ppm": {"clock_ppm": 25.0},
    "iq_imbalance": {"iq_gain_db": 0.7, "iq_phase_deg": 3.0},
    "multipath": {"multipath_taps": [1.0, 0.0, 0.0, 0.35]},
    "dc_offset": {"dc_offset": 0.05 + 0.03j},
}
# a representative modulation per family for the impairment sweep
IMPAIRMENT_MODS = [("QPSK", 20), ("16QAM", 25), ("64QAM", 30),
                   ("16APSK", 27), ("2FSK", 15), ("GMSK", 15)]

# samples-per-symbol sweep: the receiver has a sub-3-sps path and
# audio-rate telemetry runs at 40 sps and above
SPS_SWEEP = [2.5, 4.0, 8.0, 20.0, 40.0]
SPS_MODS = [("QPSK", 20), ("16QAM", 25), ("2FSK", 15)]

# bit-layer cases: FEC x interleaver x scrambler, none of which the
# shipped matrix ever enabled
BITLAYER_CASES = [
    {"name": "conv+block+pn9", "modulation": "QPSK", "snr_db": 20,
     "fec": {"family": "convolutional", "K": 7,
             "generators": (0o171, 0o133)},
     "interleaver": {"kind": "block", "rows": 8, "cols": 16},
     "scrambler": {"kind": "known_whitening", "name": "PN9-CC1101"}},
    {"name": "conv only", "modulation": "BPSK", "snr_db": 15,
     "fec": {"family": "convolutional", "K": 7,
             "generators": (0o171, 0o133)}},
    {"name": "rs+helical", "modulation": "8PSK", "snr_db": 25,
     "fec": {"family": "reed_solomon", "n": 255, "k": 223},
     "interleaver": {"kind": "helical", "rows": 8, "cols": 16, "step": 3}},
    {"name": "ldpc+conv-interleaver", "modulation": "16QAM", "snr_db": 25,
     "fec": {"family": "ldpc", "n": 256, "k": 128, "seed": 1},
     "interleaver": {"kind": "convolutional", "branches": 4, "delay": 8}},
    {"name": "scrambler only", "modulation": "QPSK", "snr_db": 20,
     "scrambler": {"kind": "known_whitening", "name": "PN9-CC1101"}},
]

# AO-73 (FUNcube-1) published ground truth (report §10.1).  The AO-40 FEC
# telemetry format designed by Phil Karn KA9Q.
AO73 = {
    "path": os.path.join(REPO, "SelfRun", "ao73.wav"),
    "sample_rate": 48000.0,
    "modulation": "BPSK",          # DBPSK; the differential branch is S7's
    "symbol_rate_hz": 1200.0,
    "carrier_hz": 1102.0,          # from the x^2 line at 2203 Hz
    "occupied_bandwidth_hz": 2000.0,
    "fec": "convolutional r=1/2 k=7 (171/133 octal), outer RS(160,128)",
    "scrambler": "CCSDS synchronous (applied before the inner code)",
    "interleaver": "80 x 65 bit block = 5200 bits",
    "frame_bits": 5200,
    "source": "Karn, 'A Proposal for a Coded AO-40 Telemetry Format'; "
              "AMSAT AO-73 mission page; Signal Identification Wiki",
}


# ----------------------------------------------------------------------
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion.

    A BER of 0.0 over 1,200 bits is not "no errors": it is "fewer than
    about 3e-3 with 95% confidence", and the difference matters when the
    number is used to justify a quality gate.  Wilson is used rather than
    the normal approximation because it stays sane at k = 0 and k = n.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def unbiased_ber(rx_bits, tx_bits, step: int, prefix: int = 2000) -> dict:
    """BER with ONE alignment, chosen from a prefix and scored on the rest.

    The shipped estimator searched alignment over a 1,200-bit window and
    kept the minimum, which is a biased estimator: it is the minimum of
    many correlated trials, so it under-reports, and it under-reports
    most where there are most trials to choose from - which is why
    128APSK could show 8.2% EVM alongside 0.0 BER (report §4.3).

    Choosing the alignment on a prefix and scoring the disjoint remainder
    removes the selection bias, because the scored bits took no part in
    the choice.  Returns the estimate with a Wilson interval and the
    number of bits it was measured over.
    """
    out = {"ber": 1.0, "n_bits": 0, "ci_low": 0.0, "ci_high": 1.0,
            "offset": None, "method": "prefix-aligned, remainder-scored"}
    if rx_bits is None or tx_bits is None:
        return out
    rx = np.asarray(rx_bits, dtype=np.uint8)
    tx = np.asarray(tx_bits, dtype=np.uint8)
    if len(rx) < 400 or len(tx) <= 400:
        return out
    head = rx[:min(prefix, len(rx) // 2)]
    if len(head) < 200:
        return out
    best_off, best_err = None, None
    for off in range(0, max(1, len(tx) - len(head)), step):
        err = int((tx[off:off + len(head)] != head).sum())
        if best_err is None or err < best_err:
            best_err, best_off = err, off
    if best_off is None:
        return out
    # score the disjoint remainder with that one alignment
    rest = rx[len(head):]
    start = best_off + len(head)
    n = min(len(rest), max(0, len(tx) - start))
    if n < 200:
        # too little left to score independently: fall back to the head,
        # and say so rather than pretending the estimate is unbiased
        n, rest, start = len(head), head, best_off
        out["method"] = "prefix-aligned, prefix-scored (stream too short)"
    errs = int((tx[start:start + n] != rest[:n]).sum())
    lo, hi = wilson_interval(errs, n)
    out.update(ber=errs / n, n_bits=int(n), ci_low=lo, ci_high=hi,
               offset=int(best_off))
    return out


def known_mod_ber(res, mod, tx_bits) -> dict:
    """BER for the true modulation, resolving only the ambiguities the
    physical layer genuinely leaves behind.

    Rotation and conjugation are real receiver ambiguities that S7
    enumerates, so trying them is legitimate; the alignment search is
    not, and that is the one this pins down.  The chosen rotation is
    reported so a reader can see how much freedom was used.
    """
    fam = MOD_FAMILY.get(mod)
    if fam in ("fsk", "gmsk"):
        k = int(math.log2(int(mod[0]))) if mod[0].isdigit() else 1
        if res.hard_bits is None or not len(res.hard_bits):
            return {"ber": 1.0, "n_bits": 0, "ci_low": 0.0, "ci_high": 1.0}
        cands = [("normal", res.hard_bits),
                 ("inverted", (1 - res.hard_bits).astype(np.uint8))]
        best = None
        for name, bits in cands:
            r = unbiased_ber(bits, tx_bits, k)
            r["mapping"] = name
            if best is None or r["ber"] < best["ber"]:
                best = r
        return best
    if res.symbols is None or not len(res.symbols):
        return {"ber": 1.0, "n_bits": 0, "ci_low": 0.0, "ci_high": 1.0}
    table, k = CONSTELLATIONS[mod]
    n_rot = symmetry_order(mod)
    best = None
    for conj in (False, True):
        base = (np.conj(res.symbols.astype(np.complex128)) if conj
                else res.symbols.astype(np.complex128))
        for r in range(n_rot):
            hard, _, _ = slice_symbols(
                base * np.exp(2j * np.pi * r / n_rot), mod, 0.05)
            got = unbiased_ber(hard, tx_bits, k)
            got["rotation"] = r
            got["conjugated"] = conj
            if best is None or got["ber"] < best["ber"]:
                best = got
            if best["ber"] < 1e-4:
                return best
    return best


# ----------------------------------------------------------------------
def _front_end(iq, cfg, sample_rate=None, real_signal=False,
               max_segments=3):
    """Run S1-S5 down the ranked segment list, as the pipeline does."""
    x, _cond = condition(iq, real_signal=real_signal)
    segs, dbg = detect_signals(x, cfg.cfar, sample_rate=sample_rate,
                               positive_only=real_signal)
    if not segs:
        return None
    last = None
    for seg in segs[:max_segments]:
        ch = channelize(x, seg)
        p = estimate_parameters(ch["samples"], cfg.params,
                                sample_rate=ch["sample_rate"],
                                analysis_band_norm=ch.get(
                                    "analysis_band_norm"))
        m = classify_modulation(ch["samples"], p, cfg.modulation)
        last = {"x": x, "segments": segs, "segment": seg, "ch": ch,
                "params": p, "modulation": m, "detection": dbg}
        if m.prediction != "UNKNOWN":
            break
    return last


EQUIV = {("QPSK", "OQPSK"), ("OQPSK", "QPSK"), ("GMSK", "2FSK"),
         ("2FSK", "GMSK"), ("OOK", "BPSK"), ("BPSK", "OOK")}


def run_case(mod, snr, seed, cfg, analyzer, full_chain, sps=8.0,
             impairment="clean", impair=None, suite="core"):
    fac = WaveformFactory(seed=seed)
    kwargs = dict(modulation=mod, sps=sps, snr_db=snr, n_frames=60)
    if mod not in ANALOG:
        kwargs.update(cfo_norm=0.004, phase_offset=0.3)
    if impair:
        kwargs["impair"] = dict(impair)
    iq, gt = fac.generate(**kwargs)
    row = {"suite": suite, "modulation": mod, "snr_db": snr, "seed": seed,
           "sps": sps, "impairment": impairment}
    t0 = time.perf_counter()

    fe = _front_end(iq, cfg)
    if fe is None:
        row.update(verdict="FAIL", reason="not detected")
        return row
    p, m, ch, seg = fe["params"], fe["modulation"], fe["ch"], fe["segment"]

    top1 = m.prediction == mod or (mod, m.prediction) in EQUIV
    top2 = top1 or any(a[0] == mod for a in m.alternatives[:2])
    rank = next((i + 1 for i, r in enumerate(m.trial_ranking)
                 if r.candidate == mod), None)
    measured = next((r.measured for r in m.trial_ranking
                     if r.candidate == mod), False)
    row.update(classified=m.prediction, class_top1=bool(top1),
               class_top2=bool(top2), rank_of_truth=rank,
               truth_was_trialled=bool(measured),
               trials_run=m.trials_run, trial_mode=m.trial_mode,
               prior_winner=m.prior_winner, trial_winner=m.trial_winner,
               segment_id=int(seg.id))

    # S4 accuracy, in the units S4 now publishes
    row["snr_state"] = p.snr_state
    row["snr_measured_db"] = p.snr_db
    # The ground truth is a full-band SNR at the original rate.  Turning
    # that into Es/N0 needs the signal's occupied bandwidth, and only for
    # a Nyquist-shaped LINEAR modulation is that one symbol rate - which
    # makes the conversion exactly 10*log10(sps).  An M-FSK signal at
    # modulation index 1 occupies about twice that, and an OFDM signal
    # occupies whatever its subcarrier count says, so quoting an "error"
    # against the linear formula for those families would be measuring
    # the formula, not the estimator.
    if p.snr_db is not None and MOD_FAMILY.get(mod) in \
            ("psk", "oqpsk", "qam", "apsk", "ask"):
        row["esn0_truth_db"] = round(snr + 10 * math.log10(sps), 1)
        row["esn0_error_db"] = round(p.snr_db - row["esn0_truth_db"], 1)
    elif p.snr_db is not None:
        row["esn0_truth_db"] = None
        row["esn0_note"] = ("Es/N0 truth is not 10*log10(sps) above the "
                            "full-band SNR for this family: its occupied "
                            "bandwidth is not one symbol rate")
    if p.symbol_rate_norm and mod not in ANALOG:
        row["symbol_rate_error_pct"] = round(
            100 * (p.symbol_rate_norm * sps - 1.0), 2)
    if p.excess_bandwidth is not None:
        row["rolloff"] = p.excess_bandwidth

    if mod == "OFDM":
        row["verdict"] = "PASS" if top1 else "FAIL"
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        return row
    if mod in ANALOG:
        res = demodulate(ch["samples"], mod, sps, cfg.demod)
        row["demod_status"] = res.demodulation_status
        row["audio_quality"] = res.lock_metrics.get(
            "audio_tonal_concentration")
        row["cfo_error"] = round(abs((res.cfo_applied_norm or 0.0) +
                                     seg.center_norm), 6)
        row["verdict"] = ("PASS" if res.demodulation_status == "GOOD"
                          else "DEGRADED" if res.demodulation_status ==
                          "DEGRADED" else "FAIL")
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        return row

    sps_est = p.samples_per_symbol if p.samples_per_symbol else sps
    if not (2 <= sps_est <= 256):
        sps_est = sps
    fam = MOD_FAMILY.get(mod)
    res = demodulate(ch["samples"], mod, sps_est, cfg.demod,
                     cfo_norm=p.carrier_offset_norm
                     if fam in ("psk", "oqpsk") else None)
    tx = np.array(gt.info_bits, dtype=np.uint8)
    ber = known_mod_ber(res, mod, tx)
    row.update(demod_status=res.demodulation_status,
               evm_percent=res.evm_percent,
               ber=round(float(ber["ber"]), 5),
               ber_ci=[round(ber["ci_low"], 5), round(ber["ci_high"], 5)],
               ber_bits=ber["n_bits"], ber_method=ber.get("method"),
               cfo_error=round(abs((res.cfo_applied_norm or 0.0) - 0.004 +
                                   seg.center_norm), 6))

    if full_chain:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            pth = os.path.join(td, "m.iq")
            np.asarray(iq, dtype=np.complex64).tofile(pth)
            fr = analyzer.analyze(pth, sample_rate=1e6,
                                  overrides={"modulation": mod})
            row["frame_detected"] = bool(fr.frames)
            row["crc_pass"] = bool(fr.frames and fr.frames.crc and
                                   fr.frames.crc["pass_fraction"] > 0.9)
            row["sync_word"] = (fr.frames.sync_word_hex
                                if fr.frames else None)
            row["sync_truth"] = gt.frame.get("sync_hex")
            row["sync_exact"] = bool(
                row["sync_word"] and row["sync_truth"] and
                row["sync_word"] == row["sync_truth"])
            pi = fr.payload_intelligence or {}
            row["payload_trust"] = (pi.get("summary") or {}).get("trust")
            truth_payload = (bytes.fromhex("".join(gt.payloads))
                             if getattr(gt, "payloads", None) else b"")
            extracted = fr.payload.data if fr.payload else b""
            row["payload_match"] = bool(
                truth_payload and extracted and
                extracted in truth_payload and
                len(extracted) >= len(truth_payload) // 2)

    good_ber = ber["ci_high"] < 0.02
    if good_ber and top1 and res.demodulation_status == "GOOD":
        row["verdict"] = "PASS"
    elif good_ber:
        row["verdict"] = "DEGRADED"
        row["reason"] = ("classification missed" if not top1 else
                         f"demod status {res.demodulation_status}")
    else:
        row["verdict"] = ("FAIL" if res.demodulation_status != "FAILED"
                          else "FAIL(gated)")
        row["reason"] = (f"BER {ber['ber']:.4f} "
                         f"[{ber['ci_low']:.4f}, {ber['ci_high']:.4f}]")
    row["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return row


def run_bitlayer_case(spec, seed, cfg, analyzer):
    """A case that actually exercises S7-S10: FEC, interleaving and
    scrambling, none of which the shipped matrix ever switched on."""
    row = {"suite": "bitlayer", "name": spec["name"], "seed": seed,
           "modulation": spec["modulation"], "snr_db": spec["snr_db"]}
    t0 = time.perf_counter()
    fac = WaveformFactory(seed=seed)
    iq, gt = fac.generate(modulation=spec["modulation"], sps=8.0,
                          snr_db=spec["snr_db"], n_frames=80,
                          cfo_norm=0.004, phase_offset=0.3,
                          fec=spec.get("fec"),
                          interleaver=spec.get("interleaver"),
                          scrambler=spec.get("scrambler"))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "m.iq")
        np.asarray(iq, dtype=np.complex64).tofile(pth)
        res = analyzer.analyze(pth, sample_rate=1e6)
    row["classified"] = res.modulation.prediction if res.modulation else None
    truth_fec = (gt.fec or {}).get("family", "none")
    truth_il = (gt.interleaver or {}).get("kind", "none")
    truth_scr = (gt.scrambler or {}).get("kind", "none")
    row["fec_truth"] = truth_fec
    row["fec_found"] = res.fec.family if res.fec else None
    row["fec_ok"] = bool(res.fec and res.fec.family == truth_fec)
    row["interleaver_truth"] = truth_il
    row["interleaver_found"] = res.interleaver.kind if res.interleaver else None
    row["interleaver_ok"] = bool(res.interleaver and
                                 res.interleaver.kind == truth_il)
    row["scrambler_truth"] = truth_scr
    row["scrambler_found"] = res.scrambler.kind if res.scrambler else None
    row["frame_detected"] = bool(res.frames)
    row["crc_pass"] = bool(res.frames and res.frames.crc and
                           res.frames.crc["pass_fraction"] > 0.9)
    truth_payload = (bytes.fromhex("".join(gt.payloads))
                     if getattr(gt, "payloads", None) else b"")
    extracted = res.payload.data if res.payload else b""
    row["payload_match"] = bool(truth_payload and extracted and
                                extracted in truth_payload)
    hits = sum([row["fec_ok"], row["interleaver_ok"],
                bool(row["crc_pass"])])
    row["verdict"] = ("PASS" if hits == 3 else
                      "DEGRADED" if hits >= 1 else "FAIL")
    row["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return row


def run_real_case(cfg, analyzer):
    """AO-73 (FUNcube-1), the one real recording in the set.

    Report §11 item 1 ranks this first: it is cheap and it makes the
    largest known failure measurable instead of unnoticed.  The
    thresholds below are deliberately loose - this is a 5.6 s clip
    holding 1.19 frames of a 5200-bit format, so full payload recovery is
    not achievable from it regardless of code quality (report §10.4) -
    but the front end either finds a 1200 baud BPSK carrier near 1102 Hz
    or it does not, and that is what is scored.
    """
    row = {"suite": "real", "name": "ao73.wav", "modulation": "BPSK",
           "ground_truth": {k: v for k, v in AO73.items() if k != "path"}}
    if not os.path.exists(AO73["path"]):
        # Absent is not the same as failing.  The recording is a large
        # binary that a fresh clone may not carry, and reporting that as
        # a FAIL would make a missing file look like a regression.
        row.update(verdict="SKIP",
                   reason=f"recording not present at {AO73['path']}")
        return row
    t0 = time.perf_counter()
    rec = load_recording(AO73["path"])
    fe = _front_end(rec.samples, cfg, sample_rate=rec.sample_rate,
                    real_signal=rec.is_real_signal, max_segments=3)
    if fe is None:
        row.update(verdict="FAIL", reason="no segment detected")
        return row
    p, seg, m, ch = fe["params"], fe["segment"], fe["modulation"], fe["ch"]
    sr = rec.sample_rate
    row["segment_hz"] = [round(seg.f_low_norm * sr, 1),
                         round(seg.f_high_norm * sr, 1)]
    row["segment_bandwidth_hz"] = round(seg.bandwidth_norm * sr, 1)
    row["carrier_hz"] = round(seg.center_norm * sr, 1)
    row["x2_carrier_hz"] = (
        round(fe["detection"]["carrier_cross_check"]["carrier_norm"] * sr, 1)
        if fe["detection"].get("carrier_cross_check", {}).get("carrier_norm")
        is not None else None)
    row["symbol_rate_hz"] = (round(p.symbol_rate_hz, 1)
                             if p.symbol_rate_hz else None)
    row["samples_per_symbol"] = (round(p.samples_per_symbol, 2)
                                 if p.samples_per_symbol else None)
    row["classified"] = m.prediction
    row["esn0_db"] = p.snr_db
    row["snr_state"] = p.snr_state

    # scoring against the published ground truth
    rate_ok = bool(row["symbol_rate_hz"] and
                   abs(row["symbol_rate_hz"] - AO73["symbol_rate_hz"]) <
                   0.05 * AO73["symbol_rate_hz"])
    # the box must bound the emission, not the whole audio band
    bw_ok = bool(row["segment_bandwidth_hz"] and
                 row["segment_bandwidth_hz"] < 4 * AO73[
                     "occupied_bandwidth_hz"])
    carrier_ok = bool(seg.f_low_norm * sr <= AO73["carrier_hz"] <=
                      seg.f_high_norm * sr)
    row.update(symbol_rate_ok=rate_ok, bandwidth_ok=bw_ok,
               carrier_in_segment=carrier_ok)

    # the bit layer, given the front end got it right
    row["reached_bit_layer"] = False
    if rate_ok and m.prediction != "UNKNOWN":
        res = demodulate(ch["samples"], "BPSK",
                         p.samples_per_symbol or 8.0, cfg.demod)
        row["demod_status"] = res.demodulation_status
        row["evm_percent"] = res.evm_percent
        row["n_bits"] = (0 if res.hard_bits is None
                         else int(len(res.hard_bits)))
        row["reached_bit_layer"] = bool(res.hard_bits is not None and
                                        len(res.hard_bits) > 256)
        if row["reached_bit_layer"]:
            from rfanalyzer.fec.detect import identify_convolutional
            from rfanalyzer.interleaving.interleavers import block_deinterleave
            bits = np.asarray(res.hard_bits, dtype=np.uint8)
            diff = np.concatenate(
                [[bits[0]], np.bitwise_xor(bits[1:], bits[:-1])]
            ).astype(np.uint8)         # AO-73 is DBPSK
            found = None
            for name, stream in (("differential", diff), ("direct", bits)):
                n = (len(stream) // 5200) * 5200
                if n < 5200:
                    continue
                de = block_deinterleave(stream[:n], 80, 65)
                hits = identify_convolutional(de, [3, 5, 7, 9])
                if hits and hits[0].get("syndrome_zero_rate", 0) > 0.6:
                    found = dict(hits[0])
                    found["stream"] = name
                    found["g1_octal"] = oct(found["g1"])
                    found["g2_octal"] = oct(found["g2"])
                    break
            row["conv_code"] = found
            row["conv_code_correct"] = bool(
                found and found.get("K") == 7 and
                {found.get("g1"), found.get("g2")} == {0o171, 0o133})

    hits = sum([rate_ok, bw_ok, carrier_ok, m.prediction != "UNKNOWN"])
    row["verdict"] = ("PASS" if hits == 4 else
                      "DEGRADED" if hits >= 2 else "FAIL")
    row["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return row


# ----------------------------------------------------------------------
def summarise(rows: list) -> dict:
    n = len(rows)
    counts = {v: sum(r.get("verdict") == v for r in rows)
              for v in ("PASS", "DEGRADED", "FAIL", "FAIL(gated)", "SKIP")}
    out = {"total": n, "pass": counts["PASS"], "degraded": counts["DEGRADED"],
           "fail": counts["FAIL"] + counts["FAIL(gated)"],
           "skipped": counts["SKIP"]}
    # per-suite and per-modulation spread across seeds
    by_mod = {}
    for r in rows:
        key = r.get("modulation") or r.get("name")
        by_mod.setdefault(key, []).append(r)
    spread = {}
    for key, rs in by_mod.items():
        verdicts = [r.get("verdict") for r in rs]
        bers = [r["ber"] for r in rs if isinstance(r.get("ber"), float)]
        errs = [r["esn0_error_db"] for r in rs
                if isinstance(r.get("esn0_error_db"), float)]
        entry = {"n": len(rs),
                 "pass": verdicts.count("PASS"),
                 "verdicts": sorted(set(v for v in verdicts if v))}
        if bers:
            entry["ber_max"] = round(max(bers), 5)
        if errs:
            entry["esn0_error_db_mean"] = round(float(np.mean(errs)), 2)
            entry["esn0_error_db_max_abs"] = round(
                float(np.max(np.abs(errs))), 2)
        spread[key] = entry
    out["per_case"] = spread
    ranks = [r["rank_of_truth"] for r in rows
             if isinstance(r.get("rank_of_truth"), int)]
    if ranks:
        out["rank_of_truth"] = {
            "top1": sum(r == 1 for r in ranks) / len(ranks),
            "top3": sum(r <= 3 for r in ranks) / len(ranks),
            "median": float(np.median(ranks)),
            "worst": int(max(ranks)),
            "never_trialled": sum(
                1 for r in rows if r.get("truth_was_trialled") is False),
        }
    out["by_suite"] = {}
    for r in rows:
        st = out["by_suite"].setdefault(r.get("suite", "core"),
                                        {"total": 0, "pass": 0})
        st["total"] += 1
        st["pass"] += (r.get("verdict") == "PASS")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="one SNR per modulation, no full-chain runs")
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of seeds per case (default 3; the shipped "
                         "harness used 1, which gives no variance estimate)")
    ap.add_argument("--suite", default="all",
                    choices=["core", "impairments", "sps", "bitlayer",
                             "real", "all"])
    ap.add_argument("--output", default="docs/validation_report.json")
    args = ap.parse_args()
    cfg = Config()
    cfg.modulation.cvnet_enabled = False       # deterministic + fast
    analyzer = RFAnalyzer(cfg, use_cache=False)
    rows = []
    suites = ([args.suite] if args.suite != "all"
              else ["core", "impairments", "sps", "bitlayer", "real"])

    def emit(row):
        rows.append(row)
        print(f"{row.get('suite','core'):11s} "
              f"{str(row.get('modulation') or row.get('name')):10s} "
              f"SNR{str(row.get('snr_db','-')):>4s} "
              f"seed{str(row.get('seed','-')):>2s} "
              f"{str(row.get('impairment',''))[:12]:12s} "
              f"class={str(row.get('classified')):9s}"
              f"{'+' if row.get('class_top1') else '-'} "
              f"rank={str(row.get('rank_of_truth','-')):>3s} "
              f"st={str(row.get('demod_status','-')):8s} "
              f"BER={str(row.get('ber','-')):8s} "
              f"CRC={str(row.get('crc_pass','-')):5s} "
              f"pay={str(row.get('payload_match','-')):5s} "
              f"-> {row['verdict']}")

    if "core" in suites:
        for mod, snrs in MATRIX.items():
            snr_list = snrs[-1:] if args.quick else snrs
            for snr in snr_list:
                for seed in range(7, 7 + args.seeds):
                    full = ((not args.quick) and snr == snrs[-1] and
                            mod not in ANALOG and mod != "OFDM" and
                            seed == 7)
                    emit(run_case(mod, snr, seed, cfg, analyzer, full))
    if "impairments" in suites:
        for mod, snr in IMPAIRMENT_MODS:
            for name, imp in IMPAIRMENTS.items():
                if name == "clean":
                    continue
                for seed in range(7, 7 + max(1, args.seeds // 2)):
                    emit(run_case(mod, snr, seed, cfg, analyzer, False,
                                  impairment=name, impair=imp,
                                  suite="impairments"))
    if "sps" in suites:
        for mod, snr in SPS_MODS:
            for sps in SPS_SWEEP:
                for seed in range(7, 7 + max(1, args.seeds // 2)):
                    emit(run_case(mod, snr, seed, cfg, analyzer, False,
                                  sps=sps, suite="sps"))
    if "bitlayer" in suites:
        for spec in BITLAYER_CASES:
            for seed in range(7, 7 + max(1, args.seeds // 2)):
                emit(run_bitlayer_case(spec, seed, cfg, analyzer))
    if "real" in suites:
        emit(run_real_case(cfg, analyzer))

    summary = summarise(rows)
    print(f"\nTOTAL: {summary['pass']} PASS, {summary['degraded']} "
          f"DEGRADED, {summary['fail']} FAIL of {summary['total']}"
          + (f" ({summary['skipped']} skipped)"
             if summary.get("skipped") else ""))
    for suite, st in summary["by_suite"].items():
        print(f"  {suite:12s} {st['pass']}/{st['total']} PASS")
    if "rank_of_truth" in summary:
        rt = summary["rank_of_truth"]
        print(f"  rank-of-truth: top1 {rt['top1']:.0%}, top3 "
              f"{rt['top3']:.0%}, worst {rt['worst']}, "
              f"{rt['never_trialled']} never trialled")
    out_path = args.output if os.path.isabs(args.output) else \
        os.path.join(REPO, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "cases": rows,
                   "configuration": {
                       "seeds": args.seeds, "quick": args.quick,
                       "suites": suites,
                       "impairments": IMPAIRMENTS,
                       "sps_sweep": SPS_SWEEP}},
                  f, indent=2, default=str)
    print("report:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
