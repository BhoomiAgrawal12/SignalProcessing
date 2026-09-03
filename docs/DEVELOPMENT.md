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
- Every estimated value travels with {value, confidence, method}.
- Never fabricate: absent knowledge is None/"unknown".
- Config over constants: tunables live in `common/config.py` and can be
  overridden by a JSON file (`configs/default.json` is a template).
- Bump `STAGE_VERSION` in `pipeline.py` (or the VERSION constants in
  stage modules) when changing an algorithm, so cached results are
  invalidated.

## Testing
- `pytest -q` runs everything; `pytest --ignore=tests/test_e2e.py` skips
  the slow end-to-end suite (about 4 minutes).
- The synthetic factory (`rfanalyzer/synth`) is the ground-truth source;
  add a factory configuration + assertions to `tests/test_e2e.py` for
  every new capability.
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

## Extension seams
- `channelize` / `demodulate` are the natural seams for a liquid-dsp or
  GNU Radio backed fast path.
- `modulation/fusion.py` accepts additional engines: return
  {"probabilities": {label: p}} and add a fusion weight.
- LDPC candidate matrices can be added to `fec/` and scored with the
  same syndrome-zero-rate metric.
