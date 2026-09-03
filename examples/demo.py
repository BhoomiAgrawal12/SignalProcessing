"""End-to-end demonstration: generate a signal with known ground truth,
analyse it blind, and compare every recovered element against the truth.

Run:  python examples/demo.py
"""
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.WARNING)

from rfanalyzer.common.config import Config
from rfanalyzer.pipeline import RFAnalyzer
from rfanalyzer.synth.factory import WaveformFactory

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    fac = WaveformFactory(seed=99)
    iq, gt = fac.generate(
        modulation="QPSK", sps=8.0, snr_db=22.0, cfo_norm=0.006,
        phase_offset=0.5,
        fec={"family": "convolutional", "K": 7, "generators": (0o171, 0o133)},
        interleaver={"kind": "block", "rows": 8, "cols": 16},
        scrambler={"kind": "known_whitening", "name": "PN9-CC1101"},
        n_frames=80)
    path = os.path.join(HERE, "demo_signal.iq")
    iq.astype(np.complex64).tofile(path)
    print(f"generated {len(iq)} samples -> {path}")
    print("ground truth: QPSK, conv K=7 (171,133), block 8x16 interleaver, "
          "PN9 whitening, 96-bit frames, sync EB90, CRC-16/CCITT-FALSE\n")

    cfg = Config()
    ckpt = os.path.join(HERE, "..", "ml", "cvnet_rf", "checkpoints",
                        "real_best.pt")
    if os.path.exists(ckpt):
        cfg.modulation.cvnet_checkpoint = os.path.abspath(ckpt)
    an = RFAnalyzer(cfg, use_cache=False)
    t0 = time.perf_counter()
    res = an.analyze(path, sample_rate=1_000_000)
    dt = time.perf_counter() - t0

    rows = [
        ("modulation", "QPSK", res.modulation.prediction),
        ("symbol rate (norm)", f"{gt.symbol_rate_norm:.5f}",
         f"{res.parameters.symbol_rate_norm:.5f}"),
        ("scrambler", "PN9-CC1101",
         res.scrambler.name if res.scrambler else None),
        ("interleaver", "block 8x16",
         f"{res.interleaver.kind} {res.interleaver.parameters.get('rows')}x"
         f"{res.interleaver.parameters.get('cols')}"
         if res.interleaver else None),
        ("FEC", "conv K=7 (0o171,0o133)",
         f"{res.fec.family} K={res.fec.parameters.get('K')} "
         f"({res.fec.parameters.get('g1_octal')},"
         f"{res.fec.parameters.get('g2_octal')})" if res.fec else None),
        ("syndrome-zero rate", "1.0 (clean)",
         res.fec.syndrome_zero_rate if res.fec else None),
        ("frame length", "96 bits",
         f"{res.frames.frame_length_bits} bits" if res.frames else None),
        ("sync word", "eb90",
         res.frames.sync_word_hex[:4] if res.frames and
         res.frames.sync_word_hex else None),
        ("CRC", "CRC-16-CCITT-FALSE",
         res.frames.crc.get("name") if res.frames and res.frames.crc
         else None),
        ("CRC pass fraction", "1.0",
         res.frames.crc.get("pass_fraction") if res.frames and
         res.frames.crc else None),
    ]
    print(f"{'element':22s} {'ground truth':28s} {'blind recovery':30s}")
    print("-" * 84)
    ok_all = True
    for name, truth, got in rows:
        ok = str(truth).split()[0].lower() in str(got).lower() or \
            str(got) == str(truth)
        ok_all &= bool(got)
        print(f"{name:22s} {truth:28s} {str(got):30s}")
    print("-" * 84)
    print(f"analysis time: {dt:.1f} s")
    if res.payload:
        print(f"payload: {len(res.payload.data)} bytes, entropy "
              f"{res.payload.entropy_bits_per_bit} bits/bit")
    out = os.path.join(HERE, "demo_result.json")
    with open(out, "w") as f:
        f.write(res.to_json(indent=2))
    print(f"full report: {out}")


if __name__ == "__main__":
    main()
