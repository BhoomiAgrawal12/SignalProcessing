"""Headerless sample-rate candidates (PS task i) against ground truth.

For realistic (sample rate, baud) pairs a QPSK burst is written as raw IQ
with no header and analysed to S4. Per case: whether the measured symbol
rate is right (within 1%), whether the true sample rate is among the
PROBABLE candidates (counted as a hit only with a right symbol rate: a
wrong rate can land on a candidate list by coincidence), and how many
candidates there are (the ambiguity; exact matches tie, so no rank). The
reported sample rate itself must stay unknown (None).

Run:  python scripts/validate_samplerate.py
Writes results/samplerate_report.json (synthetic data).
"""
import logging
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.disable(logging.INFO)

from dhwani.common.config import Config
from dhwani.pipeline import Analyzer
from dhwani.synth.factory import WaveformFactory
from provenance import write_result

# (sample rate Hz, baud): sound-card, SDR and packet-radio combinations
CASES = [(48e3, 1200), (48e3, 4800), (48e3, 9600), (96e3, 9600),
         (192e3, 38400), (250e3, 50e3), (1e6, 125e3), (2.4e6, 250e3),
         (2e6, 270833), (20e6, 1e6)]


def run_case(fs, baud, seed, analyzer):
    sps = fs / baud
    iq, _ = WaveformFactory(seed=seed).generate(
        modulation="QPSK", sps=sps, snr_db=20.0, cfo_norm=0.002,
        phase_offset=0.3, n_frames=60)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "x.iq")
        iq.astype(np.complex64).tofile(path)
        res = analyzer.analyze(path, datatype="complex64", stop_after="S5")
    p = res.parameters
    cands = [c["sample_rate_hz"] for c in (p.sample_rate_candidates if p
                                           else [])]
    rs_err = (abs(p.symbol_rate_norm_recording * sps - 1.0)
              if p and p.symbol_rate_norm_recording else None)
    rate_ok = rs_err is not None and rs_err < 0.01
    return {"sample_rate_hz": fs, "baud": baud, "sps": round(sps, 3),
            "seed": seed,
            "reported_sample_rate": res.recording_meta.get("sample_rate"),
            "candidates": cands, "n_candidates": len(cands),
            "symbol_rate_ok": rate_ok,
            "symbol_rate_rel_error": round(rs_err, 5) if rs_err is not None
            else None,
            "true_rate_found": rate_ok and fs in cands}


def main():
    analyzer = Analyzer(Config(), use_cache=False)
    rows = []
    for fs, baud in CASES:
        r = run_case(fs, baud, 3, analyzer)
        rows.append(r)
        print(f"{fs:>10,.0f} Hz {baud:>9,.0f} Bd  rate_ok="
              f"{r['symbol_rate_ok']!s:5s} found={r['true_rate_found']!s:5s}"
              f" of {r['n_candidates']} candidates", flush=True)
    summary = {"cases": len(rows),
               "symbol_rate_ok": sum(r["symbol_rate_ok"] for r in rows),
               "true_rate_found": sum(r["true_rate_found"] for r in rows),
               "mean_candidates": round(float(np.mean(
                   [r["n_candidates"] for r in rows])), 2),
               "sample_rate_ever_applied": any(
                   r["reported_sample_rate"] for r in rows)}
    print(summary)
    path = write_result("samplerate_report", {
        "settings": {"modulation": "QPSK", "snr_db": 20.0, "seed": 3,
                     "headerless": True},
        "summary": summary, "cases": rows})
    print("report:", path)


if __name__ == "__main__":
    main()
