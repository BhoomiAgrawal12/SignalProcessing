# Architecture

One engine, `dhwani.pipeline.Analyzer`, sits behind the CLI, the GUI and
every script. The browser page `web/analyze.html` runs a JavaScript port
of S0-S5 (`web/analyze.js`); `web/viewer.html` renders an exported
`analysis.json`.

## One `dhwani analyze` run

1. `cli.main` loads the config (`common/config.load_config`: defaults
   overlaid by an optional JSON; unknown keys are logged and ignored),
   points CVNet at `ml/cvnet_rf/checkpoints/<variant>_best.pt` if it
   exists, and builds `Analyzer`.
2. **S0 ingest** (`ingestion.load_recording`). `.wav` goes through
   `wav.read_wav` (own RIFF walk, `auxi` chunk for centre frequency,
   memmap). A mono WAV is real: `real_to_complex` mixes it by -fs/4,
   half-band filters and decimates by 2, so the rate becomes fs/2 and the
   centre moves by fs/4. Anything else is raw IQ, its dtype taken from
   `--datatype`, then the SigMF sidecar, then `sniffer.sniff_raw_iq`.
   The sample rate comes from the user, the WAV header, SigMF or the file
   name (GQRX / `2.4Msps` tokens, source "filename"); otherwise it stays
   `None`.
3. **S1 condition** (`conditioning.condition`): DC removal, clipping and
   dead-air report, I/Q imbalance *measured* (Gram-Schmidt correction is
   opt-in), RMS normalisation, on one 2^22-sample window. A longer file
   is scanned window by window (up to `cfar.scan_windows` = 16) through
   S1+S2 and the window with the strongest detection goes on
   (`--start-sample` pins it); `recording.analysed_start` /
   `analysed_samples` record the span and a warning states it.
4. **S2 detect** (`detection.detect_signals`): CFAR on a waterfall. The
   only cached stage: `StageCache` pickles in `cfg.cache_dir`, keyed by
   `content_hash(samples)`, `STAGE_VERSION` and the CFAR parameters.
5. **S3 channelise**, then **S4 parameters** (OBW, SNR, CFO, symbol
   rate; for headerless IQ also PROBABLE `sample_rate_candidates`).
6. **S5 modulation** (`modulation.classify_modulation`): cumulant engine
   plus receiver-trial gates, CVNet-RF fused as advisory. UNKNOWN or
   OFDM stops the run. `stop_after="S5"` ends here (`dhwani classify`).
7. **S6 demodulate** (`demod.demodulate`): symbols, hard bits, LLRs and
   a GOOD / DEGRADED / FAILED gate. FAILED, analog audio or fewer than
   256 bits stops the run.
8. **S7** ambiguity fan-out (`bits.ambiguity`, or the four FSK mappings).
9. **S8-S10** `hypothesis.bitlayer_search`: GF(2) rank pre-score, beam.
   For each survivor: whitener phase alignment or Berlekamp-Massey,
   `interleaving.identify_interleaver`, `fec.identify_fec` per
   interleaver candidate, then `_frame_and_crc` (frame length, sync,
   CRC hunt, both polarities). A confident convolutional decode without
   a CRC pass gets an RS outer-code pass (`fec.identify_outer`) and is
   re-framed. A CRC-validated chain with syndrome-zero rate > 0.95 exits
   early.
10. `_fill_result_from_best` and `_extract_payload`: frames, sync, CRC,
    payload bytes, the bit autocorrelation plot. **S11**
    `payload.analyze_payload`, capped by provenance.
11. **S12** `reporting.export_all`: `analysis.json/.csv`,
    `_payload.bin` + `.hex.txt`, `.sigmf-meta`, `.pdf` (with reportlab)
    and `.grc` (only when the sample rate is known). Each stage calls
    `trace.mark`; the resulting `pipeline_trace` drives the GUI trace
    table and the web viewer's flow animation.

`dhwani detect` is the same run with `stop_after="S2"`.

## What leaves the machine

Only `hf_hub_download` of the CVNet checkpoint (`modulation/cvnet.py`),
pinned to a revision and SHA-256, and only when ML is enabled and no
local checkpoint exists. The GUI's offline box is checked by default in
that case. The web pages make one same-origin request,
`fetch("demo/analysis.json")`, and read recordings through the File API.

## Packages

| package | role |
|---|---|
| `common/` | config dataclasses, result models (`AnalysisResult` is the JSON schema), cache, logging |
| `ingestion/` | WAV / raw IQ / SigMF loading, format sniffing, mono-to-complex |
| `conditioning/`, `detection/`, `channelization/`, `params/` | S1-S4 |
| `modulation/` | cumulant engine, CVNet adapter, fusion (S5) |
| `demod/` | per-family receivers and quality gate (S6) |
| `bits/`, `scrambling/`, `gf2/`, `interleaving/`, `fec/`, `framing/`, `hypothesis/` | the bit layer (S7-S10) |
| `payload/` | S11 payload intelligence |
| `reporting/` | S12 exports |
| `synth/` | `WaveformFactory`, the ground-truth generator behind every test, script and fixture |
| `gui/` | eight-page PyQt6 GUI running `Analyzer` on a `QThread` |
| `signatures/` | SQLite library: save and list |

## Invariants

See CLAUDE.md, "Invariants to preserve". In short: unknown stays
unknown, quality gates stop the chain, S11 is capped by provenance,
decompression is bounded, overrides are labelled, S0-S2 changes bump
`STAGE_VERSION`, untrusted inputs are escaped and never unpickled, and
the Python and JS S0-S5 engines stay in step.
