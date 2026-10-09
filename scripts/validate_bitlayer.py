"""Bit-layer ground-truth matrix: interleaver kind x FEC family x SNR.

For each case a QPSK signal with full ground truth is analysed with the
modulation pinned (analyst override), so the numbers isolate S7-S10:
  interleaver_correct  kind (and rows/cols/step, branches/delay) match
  fec_correct          family (and K / n,k / inner+outer) match
  crc_pass             a CRC validates > 90% of frames
  payload_ber          bit error rate of the recovered payload bytes
                       against the best-aligned transmitted payloads
                       (1.0 when nothing was recovered)
  frames_recovered     fraction of transmitted payloads recovered

Run:  python scripts/validate_bitlayer.py [--quick] [--seeds 1]
Writes results/bitlayer_report.json (synthetic data). Cases that never
validate explore the whole beam and can take minutes each.
"""
import argparse
import logging
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.disable(logging.INFO)

from dhwani.common.config import Config
from dhwani.pipeline import Analyzer
from dhwani.synth.factory import WaveformFactory
from provenance import write_result

CONV = {"family": "convolutional", "K": 7, "generators": (0o171, 0o133)}
RS = {"family": "reed_solomon", "n": 255, "k": 223}
FECS = {
    "none": {"family": "none"},
    "convolutional": CONV,
    "reed_solomon": RS,
    "ldpc": {"family": "ldpc", "n": 256, "k": 128, "seed": 1},
    "concatenated": {"family": "concatenated", "outer": RS, "inner": CONV},
}
INTERLEAVERS = {
    "none": {"kind": "none"},
    "block": {"kind": "block", "rows": 8, "cols": 16},
    "helical": {"kind": "helical", "rows": 8, "cols": 16, "step": 3},
    "convolutional": {"kind": "convolutional", "branches": 4, "delay": 8},
    "pseudo_random": {"kind": "pseudo_random", "period": 128},
}
SNRS = [12, 20]
IL_KEYS = {"block": ("rows", "cols"), "helical": ("rows", "cols", "step"),
           "convolutional": ("branches", "delay"),
           "pseudo_random": ()}


def interleaver_correct(res, truth: dict) -> bool:
    if not res.hypotheses:          # stopped before S8: nothing identified
        return False
    il = res.interleaver
    if il is None:
        return truth["kind"] == "none"
    if il.kind != truth["kind"]:
        return False
    if truth["kind"] == "pseudo_random":
        return il.period == truth["period"]
    return all(il.parameters.get(k) == truth[k]
               for k in IL_KEYS.get(truth["kind"], ()))


def fec_correct(res, truth: dict) -> bool:
    if not res.hypotheses:          # stopped before S8: nothing identified
        return False
    fec = res.fec
    fam = truth["family"]
    if fec is None:
        return fam == "none"
    if fec.family != fam:
        return False
    p = fec.parameters
    if fam == "convolutional":
        return p.get("K") == truth["K"]
    if fam in ("reed_solomon", "ldpc"):
        return (p.get("n"), p.get("k")) == (truth["n"], truth["k"])
    if fam == "concatenated":
        return (p["inner"].get("K") == truth["inner"]["K"] and
                (p["outer"].get("n"), p["outer"].get("k")) ==
                (truth["outer"]["n"], truth["outer"]["k"]))
    return True


def payload_scores(res, gt) -> tuple:
    """(payload_ber, frames_recovered): recovered payload bytes are cut
    into payload-sized chunks and aligned to the transmitted sequence at
    the frame offset with the fewest bit errors."""
    sent = [bytes.fromhex(p) for p in gt.payloads]
    data = res.payload.data if res.payload else b""
    size = len(sent[0]) if sent else 0
    got = [data[i:i + size] for i in range(0, len(data) - size + 1, size)]
    if not sent or not got:
        return 1.0, 0.0
    tx = np.frombuffer(b"".join(sent), dtype=np.uint8)
    rx = np.frombuffer(b"".join(got), dtype=np.uint8)
    best = None
    for k in range(len(sent)):
        n = min(len(rx), len(tx) - k * size)
        if n <= 0:
            break
        errs = int(np.unpackbits(tx[k * size:k * size + n] ^ rx[:n]).sum())
        if best is None or errs / (8 * n) < best[0]:
            best = (errs / (8 * n), n // size)
    return round(best[0], 5), round(best[1] / len(sent), 3)


def run_case(il_name, fec_name, snr, seed, analyzer):
    truth_il, truth_fec = INTERLEAVERS[il_name], FECS[fec_name]
    iq, gt = WaveformFactory(seed=seed).generate(
        modulation="QPSK", sps=8.0, snr_db=snr, cfo_norm=0.004,
        phase_offset=0.3, fec=truth_fec, interleaver=truth_il,
        scrambler={"kind": "known_whitening", "name": "PN9-CC1101"},
        n_frames=80)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "b.iq")
        iq.astype(np.complex64).tofile(path)
        # the harness knows the format; sniffing is not under test here
        res = analyzer.analyze(path, sample_rate=1e6, datatype="complex64",
                               overrides={"modulation": "QPSK"})
    ber, recovered = payload_scores(res, gt)
    return {"interleaver": il_name, "fec": fec_name, "snr_db": snr,
            "seed": seed,
            "found_interleaver": res.interleaver.label if res.interleaver
            else None,
            "found_fec": res.fec.family if res.fec else None,
            "bit_layer_ran": bool(res.hypotheses),
            "interleaver_correct": interleaver_correct(res, truth_il),
            "fec_correct": fec_correct(res, truth_fec),
            "crc_pass": bool(res.frames and res.frames.crc and
                             res.frames.crc["pass_fraction"] > 0.9),
            "payload_ber": ber, "frames_recovered": recovered,
            "elapsed_s": round(time.perf_counter() - t0, 1)}


def per_class(rows, key):
    out = {}
    for r in rows:
        c = out.setdefault(r[key], {"n": 0, "interleaver_correct": 0,
                                    "fec_correct": 0, "crc_pass": 0})
        c["n"] += 1
        for k in ("interleaver_correct", "fec_correct", "crc_pass"):
            c[k] += bool(r[k])
    from validate_matrix import wilson
    for c in out.values():
        for k in ("interleaver_correct", "fec_correct", "crc_pass"):
            c[k + "_rate"] = round(c[k] / c["n"], 3)
        c["crc_pass_ci95"] = wilson(c["crc_pass"], c["n"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="flagship SNR only")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    cfg = Config()
    cfg.modulation.cvnet_enabled = False
    analyzer = Analyzer(cfg, use_cache=False)
    snrs = SNRS[-1:] if args.quick else SNRS
    rows = []
    for il_name in INTERLEAVERS:
        for fec_name in FECS:
            for snr in snrs:
                for seed in range(11, 11 + args.seeds):
                    r = run_case(il_name, fec_name, snr, seed, analyzer)
                    rows.append(r)
                    print(f"{il_name:14s} {fec_name:14s} SNR{snr:3d} "
                          f"il={'+' if r['interleaver_correct'] else '-'} "
                          f"fec={'+' if r['fec_correct'] else '-'} "
                          f"crc={'+' if r['crc_pass'] else '-'} "
                          f"ber={r['payload_ber']:.4f} "
                          f"frames={r['frames_recovered']:.2f} "
                          f"({r['elapsed_s']} s)", flush=True)
    path = write_result("bitlayer_report", {
        "settings": {"modulation": "QPSK (pinned)", "sps": 8.0,
                     "n_frames": 80, "scrambler": "PN9-CC1101",
                     "snrs": snrs, "seeds": list(range(11, 11 + args.seeds)),
                     "cvnet": False},
        "per_interleaver": per_class(rows, "interleaver"),
        "per_fec": per_class(rows, "fec"),
        "cases": rows})
    print("report:", path)


if __name__ == "__main__":
    main()
