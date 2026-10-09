# Development

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"     # or only what you need: .[dev], .[fec], .[report], .[gui], .[ml], .[io]
```

Python 3.11 or later. Node 20 for the browser-engine test. There is no
build step: `web/` is served as-is.

## Tests

```bash
pytest                      # everything, about 2.5 minutes (CI on every push)
pytest tests/test_fec.py::test_identify_rs
python web/make-test-fixtures.py && node web/test-node.js
```

- Tests need no network and no torch; `conftest.py` disables CVNet
  unless `ml/cvnet_rf/checkpoints/real_best.pt` exists.
- `tests/test_gui_smoke.py` and `tests/test_gui_flow.py` (pytest-qt: run,
  check views, override, re-run) run offscreen with the `.[gui]` extra and
  are skipped otherwise. `python scripts/gui_screenshots.py RECORDING`
  saves every page to `results/screens/`.
- `tests/test_cvnet_safety.py::test_pickle_checkpoint_raises_without_executing`
  needs torch.
- Bug fixes start with a failing test; one roadmap item is one commit
  with its own test.

## Results

Every number in the README comes from `results/*.json`:

```bash
python scripts/benchmark.py --big
python scripts/validate_matrix.py
python scripts/validate_bitlayer.py
python scripts/validate_offair.py      # needs data/offair/manifest.json
python scripts/render_readme.py        # CI runs it with --check
```

Each file carries a provenance block (time, command, commit, CPU,
versions, data label). Never edit a result file or the README results
block by hand. `python scripts/make_demo_report.py` regenerates
`web/demo/demo_chain.wav` and `web/demo/analysis.json` (the analysis of
that same file).

## Rules worth knowing

- Any change to S0-S2 behaviour bumps `STAGE_VERSION` in `pipeline.py`
  (it keys the S2 cache).
- `web/analyze.js` ports S0-S5 and shares the modulation reference table
  with `modulation/cumulants.py`; change both together.
- Deliberate shortcuts carry `# ponytail: <ceiling>, <upgrade trigger>`
  (`// ponytail:` in JS); `/ponytail-debt` lists them.
- Untrusted inputs (recordings, SigMF sidecars, a loaded `analysis.json`)
  are escaped in HTML / PDF / GRC and never unpickled. See
  `THREAT_MODEL.md`.
- No linter, formatter or type checker is configured.
