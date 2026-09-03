"""Measured performance against the report's targets (section 10).

Run:  python scripts/benchmark.py [--big]
--big also generates a 1 GB IQ file in a temp dir to time the memmap and
first-waterfall path (needs ~1 GB free disk).
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rfanalyzer.common.config import Config
from rfanalyzer.conditioning import condition
from rfanalyzer.detection import detect_signals, compute_waterfall
from rfanalyzer.demod import demodulate
from rfanalyzer.fec.detect import identify_fec
from rfanalyzer.gf2.rank import rank_profile
from rfanalyzer.ingestion import load_recording
from rfanalyzer.params.estimators import symbol_rate
from rfanalyzer.pipeline import RFAnalyzer
from rfanalyzer.synth.factory import WaveformFactory


def t(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    fac = WaveformFactory(seed=3)
    results = []

    # detection over 10 s at 2 Msps equivalent (20 M samples)
    rng = np.random.default_rng(0)
    wide = (rng.normal(size=2_000_000) + 1j * rng.normal(size=2_000_000)).astype(np.complex64)
    sig, gt = fac.generate(modulation="QPSK", sps=8, snr_db=20, n_frames=200)
    wide[100000:100000 + len(sig)] += 3 * sig
    _, dt = t(detect_signals, wide, cfg.cfar)
    results.append(("wideband detection, 2M samples", dt, "< 2 s"))

    # symbol rate estimation
    iq, gt = fac.generate(modulation="QPSK", sps=8, snr_db=20, n_frames=400)
    _, dt = t(symbol_rate, iq)
    results.append(("symbol-rate estimation (cyclic periodogram)", dt, "1-5 s"))

    # demodulation of ~100k symbols
    iq, gt = fac.generate(modulation="QPSK", sps=8, snr_db=25, n_frames=2200)
    n_sym = len(iq) // 8
    _, dt = t(demodulate, iq, "QPSK", 8.0, cfg.demod, 0.0)
    results.append((f"demodulation of {n_sym} symbols", dt, "< 1 s per 100k"))

    # GF(2) rank scan
    bits = rng.integers(0, 2, 50000).astype(np.uint8)
    _, dt = t(rank_profile, bits, 2, 512)
    results.append(("GF(2) rank scan L=2..512, 50k bits", dt, "5-30 s"))

    # FEC identification
    from rfanalyzer.fec.conv import ConvCode, conv_encode
    coded = conv_encode(rng.integers(0, 2, 40000).astype(np.uint8),
                        ConvCode(7, (0o171, 0o133)), terminate=False)
    _, dt = t(identify_fec, coded, cfg.fec)
    results.append(("FEC identification (conv sweep + decode)", dt, "10-60 s"))

    # end-to-end clean signal
    iq, gt = fac.generate(
        modulation="QPSK", sps=8, snr_db=22, cfo_norm=0.005,
        fec={"family": "convolutional", "K": 7, "generators": (0o171, 0o133)},
        interleaver={"kind": "block", "rows": 8, "cols": 16},
        scrambler={"kind": "known_whitening", "name": "PN9-CC1101"},
        n_frames=80)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bench.iq")
        iq.astype(np.complex64).tofile(p)
        an = RFAnalyzer(cfg, use_cache=False)
        _, dt = t(an.analyze, p, 1e6)
    results.append(("end-to-end, one clean signal (full stack)", dt, "< 90 s"))

    if args.big:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "big.iq")
            block = (rng.normal(size=2_000_000) +
                     1j * rng.normal(size=2_000_000)).astype(np.complex64)
            with open(p, "wb") as f:
                for _ in range(64):           # 64 x 16 MB = 1 GB
                    block.tofile(f)
            t0 = time.perf_counter()
            rec = load_recording(p, sample_rate=2e6, datatype="complex64")
            wf = compute_waterfall(rec.samples)
            dt = time.perf_counter() - t0
            results.append(("1 GB IQ file: load + first waterfall", dt, "< 3 s"))

    print(f"\n{'operation':52s} {'measured':>10s}   target")
    print("-" * 84)
    for name, dt, target in results:
        print(f"{name:52s} {dt:9.2f}s   {target}")


if __name__ == "__main__":
    main()
