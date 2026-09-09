# Leftovers

What is not finished, why it matters, and how to tell when it is done.

Scope of this list is a solid **local** project: something an analyst can
run on their own machine against real recordings and trust the output of.
Anything that only matters for deployment, multi-user operation or scale
is collected at the bottom and is deliberately out of scope for now.

Every item says where it lives and what "done" looks like, so none of it
needs re-derivation later. Items are ordered by what would bite first.

---

## 1. Blocking: verification that has not been completed

### 1.1 The full test suite has never finished in one run

`pytest -m "not slow"` is green (197 passed, 0 failed, about 4 minutes)
and covers every stage S0 to S12. The 18 tests marked `slow` drive the
whole pipeline end to end and each takes 2 to 5 minutes; a complete
`pytest` run has not been observed finishing.

- **Where:** `tests/test_e2e.py`, `tests/test_formats_payload.py`,
  `tests/test_synth_formats.py`, `tests/test_fec_concatenated.py`,
  `tests/test_real_recording.py` (the `@pytest.mark.slow` cases).
- **Why it matters:** the fast subset exercises every stage in isolation
  but only the slow cases prove the stages compose. This is the one
  genuine hole in the current verification.
- **Done when:** `pytest -q` completes with 0 failures, once, and the
  result is recorded.

### 1.2 The Reed-Solomon sweep is the reason the slow tests are slow

Profiling puts the time in `identify_rs` -> `RSCode.syndromes`, reached
from `identify_fec` for every beam candidate. It is pure-Python GF(2^8)
arithmetic looping over 5 candidate codes x 8 bit offsets x symbol
offsets x 8 quick codewords, and it runs even when the stream carries no
block code at all.

This is **pre-existing**, confirmed by running the same case against a
clean worktree at `HEAD`. It is not a regression, but it is what makes a
full test run take tens of minutes and it will be the first thing an
analyst notices on a long recording.

- **Where:** `src/rfanalyzer/fec/detect.py:identify_rs` (cost knobs
  `quick_codewords=8`, `confirm_codewords=64`),
  `src/rfanalyzer/fec/rs.py:syndromes`.
- **Two independent fixes, either helps:**
  - vectorise `syndromes` with numpy (evaluate all `n` positions of the
    codeword polynomial at all `2t` roots as one matrix product over the
    log/antilog tables) rather than looping in Python;
  - gate the sweep: a block code imposes a length structure, so a cheap
    pre-check (does the stream's rank profile or byte-boundary
    autocorrelation show anything at 255 symbols?) can skip the sweep
    entirely for uncoded streams, which is the common case.
- **Done when:** a plain uncoded QPSK recording completes the bit layer
  in well under 10 seconds, and `pytest` finishes in single-digit
  minutes.

### 1.3 Three of the five validation suites have never been run to completion

`scripts/validate_matrix.py` implements five suites. Only `core` (27
cases, quick configuration) and `real` (1 case) have produced a finished
report. `impairments`, `sps` and `bitlayer` are written and syntax-clean
but unproven.

- **Where:** `scripts/validate_matrix.py`, suites `impairments`, `sps`,
  `bitlayer`.
- **Why it matters:** those three suites are the entire reason the
  harness was rebuilt. Phase noise, timing offset, clock ppm, IQ
  imbalance, multipath and DC offset are implemented by the synthetic
  factory and have never been exercised against the analyzer; neither has
  any sample rate other than 8 samples per symbol, nor any FEC /
  interleaver / scrambler combination.
- **Done when:** `python scripts/validate_matrix.py --suite all --seeds 3`
  completes and `docs/validation_report.json` carries all five suites.
  Expect failures the first time. That is the point of running it.

### 1.4 CI has never executed

`.github/workflows/ci.yml` is written and its YAML is well-formed, but
nothing has run it (no push happened in the session that added it).

- **Done when:** one push shows a green run on both Python 3.11 and 3.12.
- **Watch for:** the workflow installs `.[dev]`, which now pulls
  `reedsolo`. The `real` suite needs `SelfRun/ao73.wav` to be committed
  (see 5.1) or it will report SKIP rather than exercising anything.

---

## 2. Correctness gaps with a known cause

### 2.1 Es/N0 reads systematically low on linear modulations

Measured error over the 18 linear cases: **-0.9 to -4.4 dB**, always
negative, largest for the dense constellations.

- **Cause (probable, not yet confirmed):** the SNR is measured on the
  channelised segment, and the detector box tracks the -8 dB points, so
  the pulse-shaping skirts sit partly outside it. Signal power that falls
  outside the box is not counted, which biases the ratio down, and the
  denser the constellation the tighter the box the detector chooses.
- **Where:** `src/rfanalyzer/params/estimators.py:spectral_snr`, called
  from `estimate_parameters` with `band_limit` from
  `channelize()["analysis_band_norm"]`.
- **Test:** measure Es/N0 on the unchannelised signal for the same case
  and compare. If the bias disappears, integrate the signal power over
  the full channel-filter passband rather than the detector box.
- **Done when:** the error is within about +-1 dB and unbiased in sign.

### 2.2 Excess bandwidth scatters +-0.15

The raised-cosine shape fit is unbiased and monotone (true 0.2 reads
0.17 to 0.41; true 0.8 reads 0.78 to 0.86) but the spread across
modulation families is wide on 60-frame bursts. It is a large improvement
on the systematic 60% underestimate it replaced, and it is not precise.

- **Where:** `src/rfanalyzer/params/estimators.py:excess_bandwidth` and
  `_rolloff_psd`.
- **Idea worth trying:** fit the **cumulative** power integral rather
  than the PSD itself. A framed stream's spectrum is a comb whose
  envelope is the raised cosine; the comb integrates away, the integral
  is smooth and monotone in the roll-off, and the fit stops needing the
  anti-comb smoothing that currently costs resolution.
- **Done when:** +-0.05 across families, or the estimate is reported as
  `unresolvable` when it cannot reach that.

### 2.3 128QAM and 128APSK have no carrier spectral line

`E[table^4]` is 0.18 for 128QAM and exactly 0 for 128APSK, so the
symbol-domain M-power estimator has nothing to find. Both currently
depend entirely on the decision-directed frequency tracker, which needs
the amplitude scale to be right first. Through the pipeline both reach
about 2 to 3% EVM and pass, but the margin is thinner than for every
other family and has not been swept across seeds or impairments.

- **Where:** `src/rfanalyzer/demod/receiver.py:_symbol_domain_cfo`,
  `_dd_freq_track`, `_lattice_lock`.
- **Done when:** a seed x SNR x impairment sweep of those two shows the
  same failure rate as their neighbours, or the gate marks them FAILED
  honestly when it does not.

### 2.4 Concatenated FEC does not handle an interleaver between the stages

`_identify_outer_stage` runs the Reed-Solomon sweep on the inner
decoder's output. Real concatenated systems, including the AO-73 format,
put an interleaver **between** the outer block code and the inner trellis
code. S8 runs before S9, so it can only find an interleaver applied to
the transmitted stream, not one sitting inside the FEC chain.

- **Where:** `src/rfanalyzer/fec/detect.py:_identify_outer_stage`.
- **Shape of the fix:** after the inner decode, re-run interleaver
  identification on the decoded bits before the outer sweep, so the chain
  becomes inner-decode -> de-interleave -> outer-decode. The pieces all
  exist; it is a matter of calling them in that order and scoring the
  three-stage hypothesis against the two-stage one.
- **Done when:** a synthetic conv(inner) + block-interleaver +
  RS(outer) stream is identified as all three stages, and the AO-73
  recording reaches its outer RS layer rather than stopping at the
  convolutional code.

---

## 3. Implemented but never exercised

These are not known to be broken. They are known to be **untested**,
which is a different and equally reportable state.

### 3.1 The GUI has never been run

`src/rfanalyzer/gui/app.py` was modified (receiver-trial ranking table,
validity states on the parameter table, Es/N0 labelling) and only
syntax-checked. PyQt6 and pyqtgraph are not installed in `.venv`.

- **Done when:** `uv pip install --python .venv/bin/python "PyQt6"
  "pyqtgraph"`, then `rf-analyzer-gui`, load a recording, and confirm the
  modulation panel shows the trial ranking including `not measured` rows.

### 3.2 The web viewer's browser regression suite does not run

`web/viewer.js` was modified and passes `node --check`. `node
web/test-node.js` reports `0/4 passed` because its fixtures are missing.

- **Done when:** `python web/make-test-fixtures.py` (or equivalent) is
  run and `node web/test-node.js` reports 4/4, with the viewer rendering
  a report that carries `trial_ranking`, `estimates`, `detection` and
  `correlation`.

### 3.3 The ML classification engine has never been exercised

`ml/cvnet_rf/checkpoints/` is empty and `torch` is not installed, so
every measurement in this project was taken with `cvnet_enabled=False`.
The fusion weights that blend it (`cumulant` 0.5, `cvnet` 0.35) are
therefore unvalidated in combination.

- **Where:** `src/rfanalyzer/modulation/cvnet.py`, `fusion.py` fusion
  block.
- **Done when:** with a checkpoint present, the core matrix is re-run
  with the engine enabled and compared against the cumulant-only
  baseline. If it does not improve top-1 or rank-of-truth, say so and
  weight it accordingly rather than leaving it nominally in the vote.

### 3.4 Optional-dependency paths are untested

Not installed in `.venv`: `soundfile` (24-bit WAV), `sigmf` (SigMF
export), `reportlab` (PDF export), `torch` (above), `PyQt6` (above).
Each has a guarded import, so the code does not crash, but the paths
behind those guards have not been executed.

- **Done when:** `uv pip install --python .venv/bin/python -e ".[all]"`
  and the suite is run once with everything present.

### 3.5 Preamble correlation is not surfaced in the GUI or the web viewer

`--preamble` works on the CLI and the result is in `analysis.json` under
`correlation`, but neither the GUI nor `web/viewer.js` renders it.

- **Done when:** both show the hits, the transform that matched, the
  frame period and the payload range, and mark a not-significant result
  as such rather than hiding it.

### 3.6 `--save-signature` is unverified

The flag and `SignatureDB.save_from_result` exist and the pipeline now
**reads** the library (sync words feed the sync-word trimming), but the
write path was not exercised this session.

- **Done when:** an analysis is saved to the library, a second recording
  of the same emission matches it, and the trimming picks the library
  prefix over the trailing-constant-byte rule.

---

## 4. Documented capability limits

These are real limits, honestly reported by the tool today. They are
listed so nobody mistakes them for bugs, and so the cost of lifting them
is visible.

- **OFDM is parameterised, not demodulated.** FFT size and CP length are
  detected; subcarrier bits are not recovered. Lifting this means a full
  OFDM receiver (CP-based sync, FFT, per-subcarrier equalisation, pilot
  tracking), which is a project of its own.
- **LDPC is candidate-set matching only.** Known parity-check matrices
  and deterministic seeds are recognised; an arbitrary unknown H is not
  reconstructed. Blind H recovery is a research problem, not a task.
- **Pseudo-random interleavers report a period, not a permutation.**
  Recovering the permutation needs many aligned frames; the code says so
  rather than guessing.
- **Analog transmissions get audio, not a name.** FM/AM/SSB are correctly
  withheld as `UNKNOWN` by the classifier (they are not in the class
  list) and demodulated to audio. There is no positive analog
  identification path, so an analog capture is reported as "not a digital
  modulation we recognise" rather than "this is FM". Adding an analog
  branch to S5 is a contained piece of work and would improve the
  reported answer on a common real-world input.
- **Headerless raw IQ has no recoverable absolute sample rate.** By
  design. Frequencies stay in cycles per sample until an analyst pins it.

---

## 5. Housekeeping

### 5.1 `SelfRun/` is untracked

`SelfRun/ao73.wav` (535 KB) is the only real recording in the project and
`tests/test_real_recording.py` plus the `real` validation suite both
depend on it. Both skip cleanly when it is absent, so a fresh clone is
quiet rather than broken, but it also means the most valuable test does
nothing.

- **Decide:** commit it (535 KB is not unreasonable), or add a fetch
  script and document where it comes from.

### 5.2 The performance table is stale

`README.md` and `PROJECT_OVERVIEW.md` quote measurements "on an Apple
M5". Those predate every change in the current working tree and were not
taken on this machine. `scripts/benchmark.py` is unmodified and has not
been re-run.

- **Done when:** `python scripts/benchmark.py --big` is run on the target
  machine and both documents quote that, with the machine named.

### 5.3 The quoted validation numbers are from the quick configuration

`README.md` reports 27 PASS / 0 / 0 and rank-of-truth top-1 95%. That is
`--quick --seeds 1`: one SNR per modulation, one seed, core suite only.
The README says so, but it is worth replacing with a full
`--suite all --seeds 3` run once 1.3 is done.

### 5.4 Nothing is committed

The working tree carries 32 modified files and a number of new ones. It
has never been committed, so there is no bisect point if something later
regresses.

- **Suggested split, so a bisect lands somewhere useful:** measurement
  layer (S4 + models); detection and channelisation (S1 to S3); receiver
  (S6 gates, AGC, carrier); classifier (S5 trials and ranking); bit layer
  (sync-word trimming, concatenated FEC, correlation); harness, tests and
  CI; docs.

---

## 6. Out of scope for a local project

Recorded so the boundary is deliberate rather than accidental. None of
this is needed for the current goal.

- Packaging and distribution (wheel, container, entry-point install on a
  clean machine without `uv`).
- Any server, queue or multi-user mode. The engine is a library plus
  three local frontends and there is no service layer, nor should there
  be one yet.
- Streaming or real-time operation. Everything assumes a complete file on
  disk; `channelize` and the bit-layer search are batch by construction.
- Recordings that do not fit in memory. Ingestion memory-maps, but S1
  onward materialises the working array.
- GPU acceleration. The ML engine would be the only consumer and it is
  not yet validated on CPU.
- Hardware-in-the-loop capture. The project analyses recordings; it does
  not talk to an SDR.
