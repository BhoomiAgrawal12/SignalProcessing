# rf-analyzer

Automated blind analysis of `.IQ` / `.WAV` SDR recordings. The system takes an
unknown recording and works from raw RF samples to an analyst-readable
interpretation:

```
recording (.iq / .wav)
  -> format sniffing        (S0)
  -> conditioning           (S1)
  -> CFAR signal detection  (S2)
  -> channelisation         (S3)
  -> parameter estimation   (S4)
  -> modulation ID          (S5, two engines + fusion)
  -> demodulation           (S6, soft bits / LLRs)
  -> ambiguity fan-out      (S7)
  -> blind descrambling     (S7)
  -> blind de-interleaving  (S8, GF(2) rank engine)
  -> blind FEC ID + decode  (S9)
  -> framing + CRC hunting  (S10)
  -> report + exports       (S11)
```

The architecture is not a linear pipeline: stages S7-S10 run inside a
hypothesis search with beam pruning, objective evidence scoring (GF(2)
rank deficiency, FEC syndrome-zero rate, CRC pass rate) and early exit on
a validated chain. Every reported number carries a confidence and the
method that produced it; unknown things are reported as unknown.

## What it recovers, verified end to end

The bundled regression test generates a QPSK signal with a CCSDS
convolutional code (K=7, generators 171/133 octal), an 8x16 block
interleaver, PN9 whitening, 96-bit frames with a sync word and
CRC-16/CCITT, plus AWGN, carrier offset and phase offset, writes it to a
raw `.iq` file, and the analyzer blindly recovers every element of that
chain including the payload, in under 10 seconds.

## Installation

Requires Python 3.11+ (3.12 recommended).

```bash
git clone https://github.com/BhoomiAgrawal12/SignalProcessing.git
cd SignalProcessing
python -m venv .venv            # or: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -e ".[all]"         # or: uv pip install -e ".[all]"
```

Dependency groups (install only what you need):

| extra    | packages                     | needed for                       |
|----------|------------------------------|----------------------------------|
| (core)   | numpy, scipy                 | everything in S0-S10             |
| `ml`     | torch, huggingface_hub       | CVNet-RF modulation engine       |
| `gui`    | PyQt6, pyqtgraph             | desktop analyst GUI              |
| `io`     | soundfile, sigmf             | awkward WAVs, SigMF helpers      |
| `report` | reportlab                    | PDF export                       |
| `fec`    | reedsolo                     | Reed-Solomon decode              |
| `dev`    | pytest                       | test suite                       |

### CVNet-RF model

The ML modulation engine uses the verified Hugging Face model
`sohelimi/cvnet-rf` (trained on RadioML 2018.01A, 24 classes, input
(2, 1024); the `real` variant reaches 54.4% validation accuracy averaged
over -20..+30 dB SNR, which is why it is fused as an advisory engine, not
an authority). Checkpoints download automatically on first use via
`huggingface_hub`, or manually:

```bash
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('sohelimi/cvnet-rf', 'real_best.pt', \
  local_dir='ml/cvnet_rf/checkpoints')"
```

Device selection is automatic (MPS on Apple silicon, CUDA if present,
CPU otherwise); force it with `modulation.device` in the config.
Run with `--no-ml` to skip the ML engine entirely; the classical
cumulant engine covers the supported classes on its own.

## CLI

```bash
rf-analyzer detect  recording.iq --sample-rate 2000000
rf-analyzer classify recording.iq --sample-rate 2000000
rf-analyzer analyze recording.iq --sample-rate 2000000 -o out/
rf-analyzer analyze recording.wav                     # rate from WAV header
rf-analyzer analyze recording.iq --modulation BPSK    # analyst override
rf-analyzer report out/analysis.json -o out2/         # re-render exports
rf-analyzer synth test.iq --modulation QPSK --fec conv \
    --interleaver block --scrambler pn9 --snr 20      # labelled test data
rf-analyzer signatures                                # signature library
```

For a headerless raw `.iq` file the sample rate is genuinely not
recoverable from the data; without `--sample-rate` all frequencies are
reported in normalised units (cycles/sample) and become absolute the
moment you provide it.

## GUI

```bash
rf-analyzer-gui
```

Eight pages: Load and Inspect, Spectrum and Waterfall, Parameters,
Modulation, Demodulation (constellation + eye diagram), Bit Layer (rank
profile + hypothesis table), Frames and Payload (entropy field map + hex
view), Report and Export. Analysis runs on a worker thread; any decision
can be overridden and only downstream stages re-run (upstream results are
cached).

## Web report viewer

`web/` is a static site (deployable on Vercel) with a client-side viewer
for exported `analysis.json` reports: PSD, waterfall, constellation, eye
diagram, rank profile, entropy field map, verdicts and payload hex, all
rendered in the browser with no upload.

## Tests and benchmarks

```bash
pytest                       # 32 unit/integration tests + 5 end-to-end
python scripts/benchmark.py --big
```

Measured on an Apple M5 (10 cores), all report targets are met:

| operation                                | measured | target   |
|------------------------------------------|---------:|----------|
| wideband detection (2M samples)           |   0.17 s | < 2 s    |
| symbol-rate estimation                    |   0.05 s | 1-5 s    |
| demodulation of 106k symbols              |   0.63 s | < 1 s    |
| GF(2) rank scan L=2..512, 50k bits        |   0.79 s | 5-30 s   |
| FEC identification                        |   0.89 s | 10-60 s  |
| end-to-end, one clean signal (full stack) |   9.2 s  | < 90 s   |
| 1 GB IQ file to first waterfall (memmap)  |   0.09 s | < 3 s    |

## Honest limitations

- Absolute sample rate of headerless raw IQ cannot be estimated; it is
  reported unknown until the analyst pins it.
- Encrypted payloads are reported as high-entropy with no recovery
  attempted; that is a successful outcome, not a failure.
- Blind FEC/interleaver identification degrades above roughly 5% raw BER
  (fundamental to rank-based methods); improve demodulation first.
- LDPC identification is limited to candidate-set syndrome matching and
  is not yet wired into the default search.
- Pseudo-random interleavers: the period is detected, the permutation is
  reported unrecovered unless many aligned frames are available.
- OFDM is detected and parameterised (FFT size, CP length) but not
  demodulated.
- DVB-length whiteners are phase-searched over a truncated window (512
  phases of 32767) at interactive speed; the truncation is reported.
- The GNU Radio `.grc` export targets GR 3.10 block ids and is generated
  as a starting point; it is not executed or validated here.

## Technology decisions (verified, not assumed)

- **CVNet-RF**: verified on Hugging Face, architecture and checkpoint
  format read from the repository files, integrated with its own
  preprocessing. Used as an advisory engine because of its honest ~54%
  all-SNR accuracy.
- **GNU Radio**: not installed on the dev machine and deliberately kept
  optional; the core is pure NumPy/SciPy with a clean seam
  (`channelize`, `demodulate`) where a GR-backed implementation can be
  slotted in, plus `.grc` export.
- **DSpectrum / DSpectrumGUI** (tresacton, the tool "D-spectrum" refers
  to): verified to exist but deprecated, Ruby-based and aimed at simple
  OOK/PWM ISM remotes; unsuitable here, so the reverse-engineering
  functionality is implemented natively (bit layer engine).
- **TorchSig**: not used; our synthetic factory generates the labels
  TorchSig cannot (FEC, interleaver, scrambler, frame ground truth).
- **liquid-dsp / AFF3CT / libfec**: not required to meet the performance
  targets (see benchmarks); candidates for a future fast path behind the
  existing interfaces.
- **reedsolo** (RS decode), **sigmf**, **soundfile**, **reportlab**:
  verified and used.

See `docs/ARCHITECTURE.md`, `docs/ALGORITHMS.md` and
`docs/DEVELOPMENT.md` for the deep dives.
