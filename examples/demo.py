"""End-to-end demonstration: generate a signal with known ground truth,
analyse it blind, and compare every recovered element against the truth.

Run:  python examples/demo.py
"""
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.WARNING)

from dhwani.common.config import Config
from dhwani.pipeline import Analyzer
from dhwani.synth.factory import DEMO_CHAIN, WaveformFactory

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    fac = WaveformFactory(seed=99)
    iq, gt = fac.generate(**DEMO_CHAIN)
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
    an = Analyzer(cfg, use_cache=False)
    t0 = time.perf_counter()
    res = an.analyze(path, sample_rate=1_000_000)
    dt = time.perf_counter() - t0

    il, fec, fr = res.interleaver, res.fec, res.frames
    il_p = il.parameters if il else {}
    fec_p = fec.parameters if fec else {}
    crc = (fr.crc or {}) if fr else {}
    sync = (fr.sync_word_hex or "") if fr else ""
    rs = res.parameters.symbol_rate_norm if res.parameters else None
    # (element, ground truth, blind recovery, pass?)
    rows = [
        ("modulation", "QPSK", res.modulation.prediction,
         res.modulation.prediction == "QPSK"),
        ("symbol rate (norm)", f"{gt.symbol_rate_norm:.5f}",
         f"{rs:.5f}" if rs else None,
         bool(rs) and abs(rs / gt.symbol_rate_norm - 1) < 0.01),
        ("scrambler", "PN9-CC1101",
         res.scrambler.name if res.scrambler else None,
         bool(res.scrambler) and res.scrambler.name == "PN9-CC1101"),
        ("interleaver", "block 8x16",
         f"{il.kind} {il_p.get('rows')}x{il_p.get('cols')}" if il else None,
         bool(il) and (il.kind, il_p.get("rows"), il_p.get("cols"))
         == ("block", 8, 16)),
        ("FEC", "conv K=7 (0o171,0o133)",
         f"{fec.family} K={fec_p.get('K')} ({fec_p.get('g1_octal')},"
         f"{fec_p.get('g2_octal')})" if fec else None,
         bool(fec) and fec.family == "convolutional" and fec_p.get("K") == 7
         and {fec_p.get("g1_octal"), fec_p.get("g2_octal")}
         == {"0o171", "0o133"}),
        ("syndrome-zero rate", ">= 0.95",
         fec.syndrome_zero_rate if fec else None,
         bool(fec) and (fec.syndrome_zero_rate or 0) >= 0.95),
        ("frame length", "96 bits",
         f"{fr.frame_length_bits} bits" if fr else None,
         bool(fr) and fr.frame_length_bits == 96),
        ("sync word", "eb90", sync[:4] or None, sync.startswith("eb90")),
        ("CRC", "CRC-16-CCITT-FALSE", crc.get("name"),
         crc.get("name") == "CRC-16-CCITT-FALSE"),
        ("CRC pass fraction", ">= 0.95", crc.get("pass_fraction"),
         (crc.get("pass_fraction") or 0) >= 0.95),
    ]
    print(f"{'element':22s} {'ground truth':26s} {'blind recovery':30s} result")
    print("-" * 86)
    for name, truth, got, ok in rows:
        print(f"{name:22s} {truth:26s} {str(got):30s} "
              f"{'PASS' if ok else 'FAIL'}")
    print("-" * 86)
    n_fail = sum(not ok for *_, ok in rows)
    print(f"analysis time: {dt:.1f} s")
    if res.payload:
        print(f"payload: {len(res.payload.data)} bytes, entropy "
              f"{res.payload.entropy_bits_per_bit} bits/bit")
    out = os.path.join(HERE, "demo_result.json")
    with open(out, "w") as f:
        f.write(res.to_json(indent=2))
    print(f"full report: {out}")
    print(f"{len(rows) - n_fail}/{len(rows)} PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
