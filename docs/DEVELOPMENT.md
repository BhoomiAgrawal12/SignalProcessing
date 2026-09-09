# Development

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[all]"
.venv/bin/pytest -q
```

## Conventions
- Type hints and docstrings on public APIs; comments state constraints,
  not narration. No em dashes in source files.
- Every estimated value travels with {value, confidence, method, state}
  (`common/models.Estimate`). `state` is `valid`, `saturated` or
  `unresolvable`; publish it and check it before comparing a value
  against a threshold.
- Never fabricate: absent knowledge is None/"unknown".
- Config over constants: tunables live in `common/config.py` and can be
  overridden by a JSON file (`configs/default.json` is a template).
- Bump `STAGE_VERSION` in `pipeline.py` (or the VERSION constants in
  stage modules) when changing an algorithm, so cached results are
  invalidated.

## Testing
- `pytest -q` runs everything and takes tens of minutes: a dozen tests
  drive the whole pipeline, and the bit-layer search on an uncoded
  stream spends most of its time in the Reed-Solomon alignment sweep.
  Those are marked `slow`, so `pytest -m "not slow"` gives a full signal
  on every stage in a couple of minutes and is what to run while
  iterating.
- CI (`.github/workflows/ci.yml`) runs the suite on 3.11 and 3.12 for
  every push and pull request, and the ground-truth matrix plus the real
  recording on `main`.
- The synthetic factory (`rfanalyzer/synth`) is the ground-truth source;
  add a factory configuration + assertions to `tests/test_e2e.py` for
  every new capability.
- `scripts/validate_matrix.py` is the regression backbone. Suites:
  `core` (every modulation x SNR x seed), `impairments` (phase noise,
  timing offset, clock ppm, IQ imbalance, multipath, DC offset), `sps`
  (2.5 to 40 samples per symbol), `bitlayer` (FEC x interleaver x
  scrambler) and `real` (AO-73 against published ground truth). Use
  several seeds: one seed gives no variance estimate.
- BER is scored with ONE alignment chosen from a prefix and applied to
  the disjoint remainder, reported with a Wilson interval. Searching
  alignment and keeping the minimum is a biased estimator and produced
  an 8.2% EVM alongside a 0.0 BER.
- `python scripts/benchmark.py --big` reproduces the performance table.

## Adding a modulation
1. Add the constellation + Gray mapping to `demod/constellations.py`
   (the cumulant references derive automatically).
2. Add the label to `ModulationConfig.classes` and, if applicable, to
   `CVNET_MAP` in `modulation/fusion.py`.
3. Extend `synth/factory.py` if the factory should generate it.
4. Add a round-trip case to the classifier and demod tests.

## Adding a FEC family / interleaver / whitener
- FEC: implement identify + decode in `fec/`, register in
  `identify_fec`; score with a syndrome-style objective test.
- Interleaver: implement forward + exact inverse in
  `interleaving/interleavers.py`, add a hypothesis sweep in
  `identify_interleaver`.
- Whitener: one entry in `scrambling/lfsr.py:KNOWN_WHITENERS` is enough;
  phase alignment picks it up automatically.

## Adding a known-pattern search
- `bits/correlate.py` is standalone: `correlate_pattern(bits, "A6 3C 91")`
  works on any bit array. The pipeline threads it through `--preamble`
  and searches every stage's bits.
- New ambiguity transforms go in `_transforms`; a transform must be one
  the physical layer genuinely leaves behind, because each one searched
  multiplies the number of alignments and therefore raises the
  significance threshold for everything.

## Extension seams
- `channelize` / `demodulate` are the natural seams for a liquid-dsp or
  GNU Radio backed fast path.
- `modulation/fusion.py` accepts additional engines: return
  {"probabilities": {label: p}} and add a fusion weight.
- LDPC candidate matrices can be added to `fec/` and scored with the
  same syndrome-zero-rate metric.
