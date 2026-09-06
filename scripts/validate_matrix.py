"""Ground-truth validation matrix across every supported modulation.

For each modulation x SNR (x impairments), a signal with full ground
truth is generated and measured on:
  1. blind classification (S5) - top-1 and top-2
  2. demodulation with the true modulation (S6) - status, EVM, BER
  3. the fully blind chain (S0-S11) - frame detection, CRC, payload trust

Verdicts per case:
  PASS      blind class top-1 correct AND BER < 2%
  DEGRADED  BER < 2% but classification missed top-1, or demod DEGRADED
  FAIL      unreliable bits (and honestly gated) or chain failure

Run:  .venv/bin/python scripts/validate_matrix.py [--quick] [--seeds 1]
"""
import argparse
import json
import logging
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
from rfanalyzer.demod.constellations import CONSTELLATIONS, MOD_FAMILY, slice_symbols
from rfanalyzer.pipeline import RFAnalyzer
from rfanalyzer.synth.factory import WaveformFactory

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


def mid_window_ber(rx_bits, tx_bits, step, win=1200):
    if rx_bits is None or len(rx_bits) < 400:
        return 1.0
    start = (max(0, len(rx_bits) // 4) // step) * step
    seg = rx_bits[start:start + win]
    if len(seg) < 400 or len(tx_bits) <= len(seg):
        return 1.0
    best = 1.0
    for off in range(0, len(tx_bits) - len(seg), step):
        b = float((tx_bits[off:off + len(seg)] != seg).mean())
        if b < best:
            best = b
        if best < 0.003:
            break
    return best


def known_mod_ber(res, mod, tx_bits):
    fam = MOD_FAMILY.get(mod)
    if fam in ("fsk", "gmsk"):
        k = 2 if mod == "4FSK" else 1
        if res.hard_bits is None or not len(res.hard_bits):
            return 1.0
        return min(mid_window_ber(res.hard_bits, tx_bits, k),
                   mid_window_ber((1 - res.hard_bits).astype(np.uint8),
                                  tx_bits, k))
    if res.symbols is None or not len(res.symbols):
        return 1.0
    from rfanalyzer.demod.constellations import symmetry_order
    table, k = CONSTELLATIONS[mod]
    n_rot = symmetry_order(mod)
    best = 1.0
    for conj in (False, True):
        base = np.conj(res.symbols.astype(np.complex128)) if conj \
            else res.symbols.astype(np.complex128)
        for r in range(n_rot):
            hard, _, _ = slice_symbols(base * np.exp(2j * np.pi * r / n_rot),
                                       mod, 0.05)
            best = min(best, mid_window_ber(hard, tx_bits, k))
            if best < 0.003:
                return best
    return best


def run_case(mod, snr, seed, cfg, analyzer, full_chain):
    fac = WaveformFactory(seed=seed)
    kwargs = dict(modulation=mod, sps=8.0, snr_db=snr, n_frames=60)
    if mod not in ANALOG:
        kwargs.update(cfo_norm=0.004, phase_offset=0.3)
    iq, gt = fac.generate(**kwargs)
    row = {"modulation": mod, "snr_db": snr, "seed": seed}
    t0 = time.perf_counter()

    x, _ = condition(iq)
    segs, _ = detect_signals(x, cfg.cfar)
    if not segs:
        row.update(verdict="FAIL", reason="not detected")
        return row
    ch = channelize(x, segs[0])
    p = estimate_parameters(ch["samples"], cfg.params)

    # 1. blind classification
    m = classify_modulation(ch["samples"], p, cfg.modulation)
    # informationally inseparable pairs: OQPSK is QPSK with a stagger the
    # classifier cannot see pre-carrier; OOK collapses onto bipolar BPSK
    # once S1 removes the DC term; GMSK/2FSK share the discriminator view
    equiv = {("QPSK", "OQPSK"), ("OQPSK", "QPSK"), ("GMSK", "2FSK"),
             ("2FSK", "GMSK"), ("OOK", "BPSK"), ("BPSK", "OOK")}
    top1 = m.prediction == mod or (mod, m.prediction) in equiv
    top2 = top1 or any(a[0] == mod for a in m.alternatives[:2])
    row["classified"] = m.prediction
    row["class_top1"] = bool(top1)
    row["class_top2"] = bool(top2)

    # analog and OFDM: classification + (for analog) demod status is the test
    if mod == "OFDM":
        row["verdict"] = "PASS" if top1 else "FAIL"
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        return row
    if mod in ANALOG:
        res = demodulate(ch["samples"], mod, 8.0, cfg.demod)
        row["demod_status"] = res.demodulation_status
        row["audio_quality"] = res.lock_metrics.get(
            "audio_tonal_concentration")
        row["verdict"] = ("PASS" if res.demodulation_status == "GOOD"
                          else "DEGRADED" if res.demodulation_status ==
                          "DEGRADED" else "FAIL")
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        return row

    # 2. demodulation with the true modulation
    sps_est = 1.0 / p.symbol_rate_norm if p.symbol_rate_norm else 8.0
    if not (2 <= sps_est <= 64):
        sps_est = 8.0
    fam = MOD_FAMILY.get(mod)
    res = demodulate(ch["samples"], mod, sps_est, cfg.demod,
                     cfo_norm=p.carrier_offset_norm
                     if fam in ("psk", "oqpsk") else None)
    tx = np.array(gt.info_bits, dtype=np.uint8)
    ber = known_mod_ber(res, mod, tx)
    row.update(demod_status=res.demodulation_status,
               evm_percent=res.evm_percent, ber=round(float(ber), 4),
               cfo_error=abs((res.cfo_applied_norm or 0) -
                             0.004 + segs[0].center_norm))

    # 3. full blind chain (frames/CRC/payload) on the flagship SNR only
    if full_chain:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            pth = os.path.join(td, "m.iq")
            iq.astype(np.complex64).tofile(pth)
            fr = analyzer.analyze(pth, sample_rate=1e6,
                                  overrides={"modulation": mod})
            row["frame_detected"] = bool(fr.frames)
            row["crc_pass"] = bool(fr.frames and fr.frames.crc and
                                   fr.frames.crc["pass_fraction"] > 0.9)
            pi = fr.payload_intelligence or {}
            row["payload_trust"] = (pi.get("summary") or {}).get("trust")

    good_ber = ber < 0.02
    gate_honest = (res.demodulation_status == "FAILED") == (ber > 0.1) or \
        res.demodulation_status in ("GOOD", "DEGRADED")
    if good_ber and top1 and res.demodulation_status == "GOOD":
        row["verdict"] = "PASS"
    elif good_ber:
        row["verdict"] = "DEGRADED"
        row["reason"] = ("classification missed" if not top1 else
                         f"demod status {res.demodulation_status}")
    else:
        row["verdict"] = "FAIL" if res.demodulation_status != "FAILED" \
            else "FAIL(gated)"
        row["reason"] = f"BER {ber:.3f}"
    row["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="one SNR per modulation, no full-chain runs")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--output", default="docs/validation_report.json")
    args = ap.parse_args()
    cfg = Config()
    cfg.modulation.cvnet_enabled = False       # deterministic + fast
    analyzer = RFAnalyzer(cfg, use_cache=False)
    rows = []
    for mod, snrs in MATRIX.items():
        snr_list = snrs[-1:] if args.quick else snrs
        for snr in snr_list:
            for seed in range(7, 7 + args.seeds):
                full = (not args.quick) and snr == snrs[-1] and \
                    mod not in ANALOG and mod != "OFDM"
                row = run_case(mod, snr, seed, cfg, analyzer, full)
                rows.append(row)
                print(f"{mod:10s} SNR{snr:3.0f} seed{seed}: "
                      f"class={str(row.get('classified')):9s}"
                      f"{'+' if row.get('class_top1') else '-'} "
                      f"status={str(row.get('demod_status', '-')):8s} "
                      f"BER={str(row.get('ber', '-')):8s} "
                      f"CRC={str(row.get('crc_pass', '-')):5s} "
                      f"trust={str(row.get('payload_trust', '-')):20s} "
                      f"-> {row['verdict']}")
    n = len(rows)
    n_pass = sum(r["verdict"] == "PASS" for r in rows)
    n_deg = sum(r["verdict"] == "DEGRADED" for r in rows)
    n_fail = n - n_pass - n_deg
    summary = {"total": n, "pass": n_pass, "degraded": n_deg, "fail": n_fail}
    print(f"\nTOTAL: {n_pass} PASS, {n_deg} DEGRADED, {n_fail} FAIL of {n}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "cases": rows}, f, indent=2,
                  default=str)
    print("report:", args.output)


if __name__ == "__main__":
    main()
