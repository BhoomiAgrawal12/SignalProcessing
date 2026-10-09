"""Dhwani command-line interface.

Shares the Analyzer engine with the GUI - no duplicated logic.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _add_common(p):
    p.add_argument("file", help="input .wav / .iq / .sigmf-data file")
    p.add_argument("--sample-rate", type=float, default=None,
                   help="sample rate in Hz (required for absolute units on "
                        "headerless raw IQ)")
    p.add_argument("--center-frequency", type=float, default=None)
    p.add_argument("--datatype", choices=["complex64", "float32", "int16",
                                          "int8", "uint8"], default=None,
                   help="raw IQ sample format (skips sniffing)")
    p.add_argument("--start-sample", type=int, default=None,
                   help="analyse the 2^22-sample window starting here "
                        "(default: the window with the strongest signal)")
    p.add_argument("--config", default=None, help="JSON config file")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--no-cache", action="store_true")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="dhwani",
        description="Automated blind analysis of .IQ/.WAV SDR recordings")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="full S0-S11 analysis")
    _add_common(a)
    a.add_argument("--signal", type=int, default=0,
                   help="index of the detected signal to analyse")
    a.add_argument("--modulation", default=None,
                   help="analyst override for the modulation")
    a.add_argument("--symbol-rate-norm", type=float, default=None,
                   help="analyst override, symbols per recorded sample")
    a.add_argument("--output", "-o", default="rf_analysis",
                   help="output directory for exports")
    a.add_argument("--json", action="store_true",
                   help="print the result JSON to stdout")
    a.add_argument("--no-ml", action="store_true",
                   help="disable the CVNet-RF engine")
    a.add_argument("--save-signature", default=None, metavar="NAME",
                   help="store the solved signal in the signature library")

    d = sub.add_parser("detect", help="S0-S2 only: list detected signals")
    _add_common(d)
    d.add_argument("--json", action="store_true")

    c = sub.add_parser("classify", help="S0-S5 only: modulation classification")
    _add_common(c)
    c.add_argument("--signal", type=int, default=0)
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-ml", action="store_true")

    r = sub.add_parser("report", help="re-render exports from a saved JSON")
    r.add_argument("json_file")
    r.add_argument("--output", "-o", default="rf_analysis")

    s = sub.add_parser(
        "synth", help="generate a labelled synthetic recording",
        description="Generate a synthetic recording with full ground truth. "
        "The output format follows the file extension: "
        ".iq/.bin/.raw = raw interleaved complex64; "
        ".wav = stereo WAV (left=I, right=Q, float32 by default, needs "
        "--sample-rate); "
        ".sigmf = SigMF pair (<base>.sigmf-data cf32_le + <base>.sigmf-meta, "
        "ground truth in the dhwani: extension namespace). "
        "Every format also writes <base>_truth.json.")
    s.add_argument("output", help="output path; extension selects the format "
                   "(.iq | .wav | .sigmf)")
    s.add_argument("--modulation", default="QPSK")
    s.add_argument("--snr", type=float, default=20.0)
    s.add_argument("--sps", type=float, default=8.0)
    s.add_argument("--fec", choices=["none", "conv", "rs", "ldpc", "concat"],
                   default="none",
                   help="concat = RS(255,223) outer + conv K=7 inner")
    s.add_argument("--interleaver", default="none",
                   choices=["none", "block", "diagonal", "helical",
                            "convolutional", "ieee80211", "pseudo_random"],
                   help="diagonal = helical (same interleaver)")
    s.add_argument("--scrambler", choices=["none", "pn9"], default="none")
    s.add_argument("--frames", type=int, default=80)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--sample-rate", type=float, default=None,
                   help="sample rate in Hz (required for .wav, recommended "
                        "for .sigmf; recorded in the header/metadata)")
    s.add_argument("--centre-frequency", "--center-frequency",
                   dest="centre_frequency", type=float, default=None,
                   help="centre frequency in Hz (recorded in SigMF captures)")
    s.add_argument("--payload", choices=["random", "text"], default="random",
                   help="frame payload content: random printable bytes or "
                        "readable telemetry-style text (nice for the S11 demo)")
    s.add_argument("--wav-bits", type=int, choices=[16, 32], default=32,
                   help="WAV sample format: 32 = IEEE float32 (default, "
                        "exact), 16 = PCM16")

    sub.add_parser("signatures", help="list the signature library")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False)
                        else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    if args.cmd == "synth":
        return _cmd_synth(args)
    if args.cmd == "report":
        return _cmd_report(args)
    if args.cmd == "signatures":
        return _cmd_signatures()

    from .common.config import load_config
    from .pipeline import Analyzer
    cfg = load_config(getattr(args, "config", None))
    local_ckpt = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        "ml", "cvnet_rf", "checkpoints", f"{cfg.modulation.cvnet_variant}_best.pt")
    if os.path.exists(local_ckpt) and not cfg.modulation.cvnet_checkpoint:
        cfg.modulation.cvnet_checkpoint = local_ckpt
    an = Analyzer(cfg, use_cache=not args.no_cache)

    if args.cmd == "detect":
        return _detect_only(an, args)
    if args.cmd == "classify":
        return _classify_only(an, args)

    overrides = {}
    if args.modulation:
        overrides["modulation"] = args.modulation
    if args.symbol_rate_norm:
        overrides["symbol_rate_norm"] = args.symbol_rate_norm
    result = an.analyze(args.file, sample_rate=args.sample_rate,
                        center_frequency=args.center_frequency,
                        datatype=args.datatype, signal_index=args.signal,
                        overrides=overrides, no_ml=args.no_ml,
                        start_sample=args.start_sample)
    from .reporting import export_all
    written = export_all(result, args.output)
    _print_summary(result)
    print("\nExports written:")
    for k, v in written.items():
        print(f"  {k:8s} {v}")
    if args.save_signature:
        from .signatures import SignatureDB
        db = SignatureDB(cfg.signature_db)
        sid = db.save_from_result(result, args.save_signature)
        print(f"signature saved as id {sid}")
    if args.json:
        print(result.to_json(indent=2))
    return 0


def _detect_only(an, args):
    res = an.analyze(args.file, sample_rate=args.sample_rate,
                     center_frequency=args.center_frequency,
                     datatype=args.datatype, stop_after="S2",
                     start_sample=args.start_sample)
    if args.json:
        from .common.models import to_json
        print(to_json({"recording": res.recording_meta,
                       "signals": [s.to_dict() for s in res.segments]},
                      indent=2))
        return 0
    print(f"file: {args.file}")
    for w in res.warnings:
        print(f"  warning: {w}")
    print(f"detected {len(res.segments)} signal(s):")
    for s in res.segments:
        line = (f"  [{s.id}] centre {s.center_norm:+.4f} "
                f"bw {s.bandwidth_norm:.4f} (normalised)  "
                f"SNR {s.snr_db:.1f} dB")
        if s.sample_rate:
            line += (f"  | {s.absolute('center'):,.0f} Hz, "
                     f"bw {s.absolute('bandwidth'):,.0f} Hz")
        print(line)
    return 0


def _classify_only(an, args):
    res = an.analyze(args.file, sample_rate=args.sample_rate,
                     center_frequency=args.center_frequency,
                     datatype=args.datatype, signal_index=args.signal,
                     no_ml=args.no_ml, stop_after="S5",
                     start_sample=args.start_sample)
    if not res.segments:
        print("no signals detected")
        return 1
    p, m = res.parameters, res.modulation
    if args.json:
        print(json.dumps({"parameters": p.to_dict(),
                          "modulation": m.to_dict()}, indent=2, default=str))
        return 0
    print(f"signal {res.selected_segment['id']}: SNR {p.snr_db} dB, "
          f"Rs {p.symbol_rate_norm_recording} symbols/recorded sample")
    print(f"modulation: {m.prediction}  confidence {m.confidence}")
    for label, prob in m.alternatives:
        print(f"    {label:8s} {prob:.3f}")
    if not m.classifier_agreement:
        print("  NOTE: engines disagree - inspect before trusting")
    return 0


def _cmd_report(args):
    import types
    with open(args.json_file) as f:
        data = json.load(f)
    shim = types.SimpleNamespace(
        to_dict=lambda: data, to_json=lambda **kw: json.dumps(data, **kw),
        recording_meta=data.get("recording", {}),
        payload=types.SimpleNamespace(
            data=bytes.fromhex((data.get("payload") or {}).get("hex", "")))
        if data.get("payload") else None)
    from .reporting import export_all
    written = export_all(shim, args.output)
    for k, v in written.items():
        print(f"  {k:8s} {v}")
    return 0


def _cmd_synth(args):
    from .synth.factory import WaveformFactory
    from .synth.writers import write_recording
    fac = WaveformFactory(seed=args.seed)
    fec = {"family": "none"}
    if args.fec == "conv":
        fec = {"family": "convolutional", "K": 7, "generators": (0o171, 0o133)}
    elif args.fec == "rs":
        fec = {"family": "reed_solomon", "n": 255, "k": 223}
    elif args.fec == "ldpc":
        fec = {"family": "ldpc", "n": 256, "k": 128, "seed": 1}
    elif args.fec == "concat":
        fec = {"family": "concatenated",
               "outer": {"family": "reed_solomon", "n": 255, "k": 223},
               "inner": {"family": "convolutional", "K": 7,
                         "generators": (0o171, 0o133)}}
    il = {"kind": "none"}
    if args.interleaver == "block":
        il = {"kind": "block", "rows": 8, "cols": 16}
    elif args.interleaver in ("diagonal", "helical"):
        il = {"kind": "helical", "rows": 8, "cols": 16, "step": 3}
    elif args.interleaver == "convolutional":
        il = {"kind": "convolutional", "branches": 4, "delay": 8}
    elif args.interleaver == "ieee80211":
        il = {"kind": "ieee80211", "ncbps": 192, "nbpsc": 4}
    elif args.interleaver == "pseudo_random":
        il = {"kind": "pseudo_random", "period": 128}
    scr = {"kind": "none"}
    if args.scrambler == "pn9":
        scr = {"kind": "known_whitening", "name": "PN9-CC1101"}
    iq, gt = fac.generate(modulation=args.modulation, sps=args.sps,
                          snr_db=args.snr, cfo_norm=0.005, phase_offset=0.4,
                          fec=fec, interleaver=il, scrambler=scr,
                          n_frames=args.frames, payload_mode=args.payload)
    written = write_recording(iq, gt, args.output,
                              sample_rate=args.sample_rate,
                              center_frequency=args.centre_frequency,
                              wav_bits=args.wav_bits)
    print(f"generated {len(iq)} complex64 samples "
          f"({args.modulation}, SNR {args.snr} dB, FEC {args.fec})")
    for kind, path in written.items():
        print(f"  {kind:6s} {path}")
    return 0


def _cmd_signatures():
    from .common.config import load_config
    from .signatures import SignatureDB
    cfg = load_config()
    db = SignatureDB(cfg.signature_db)
    sigs = db.list_all()
    if not sigs:
        print("signature library is empty")
    for s in sigs:
        print(f"  [{s['id']}] {s['name']}: {s['modulation']} "
              f"sync={s['sync_word_hex']}")
    return 0


def _print_summary(result):
    r = result
    print("\n=== Dhwani result ===")
    print(f"run: {r.run_id}")
    for w in r.warnings:
        print(f"warning: {w}")
    if r.modulation:
        print(f"modulation: {r.modulation.prediction} "
              f"(confidence {r.modulation.confidence})")
    if r.parameters:
        p = r.parameters
        print(f"SNR {p.snr_db} dB | Rs {p.symbol_rate_norm_recording} "
              "symbols/recorded sample"
              + (f" = {p.symbol_rate_hz:,.0f} Bd" if p.symbol_rate_hz else ""))
        for c in p.sample_rate_candidates[:4]:
            print(f"sample-rate candidate (PROBABLE, not applied): "
                  f"{c['sample_rate_hz']:,.0f} Hz -> "
                  f"{c['standard_symbol_rate']:,.0f} Bd")
    if r.scrambler:
        print(f"scrambler: {r.scrambler.kind} {r.scrambler.name}")
    if r.interleaver:
        print(f"interleaver: {r.interleaver.label} {r.interleaver.parameters}")
    if r.fec:
        print(f"FEC: {r.fec.family} {r.fec.parameters} "
              f"syndrome-zero-rate={r.fec.syndrome_zero_rate}")
    if r.frames:
        crc = r.frames.crc
        print(f"frames: {r.frames.frame_length_bits} bits, "
              f"sync {r.frames.sync_word_hex}, "
              f"CRC {crc['name'] + ' pass ' + str(crc['pass_fraction']) if crc else 'none found'}")
    if r.payload:
        print(f"payload: {len(r.payload.data)} bytes, "
              f"entropy {r.payload.entropy_bits_per_bit} b/b"
              + (" (likely encrypted)" if r.payload.likely_encrypted else ""))


if __name__ == "__main__":
    sys.exit(main())
