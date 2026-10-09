"""Regenerate web/demo/analysis.json (the viewer's ?demo=1 report) from a
seeded synthetic recording: the examples/demo.py chain with a text
payload, so the S11 stage has something to show.

Run:  python scripts/make_demo_report.py
"""
import logging
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
logging.disable(logging.INFO)

from dhwani.common.config import Config
from dhwani.pipeline import Analyzer
from dhwani.reporting.report import export_json
from dhwani.synth.factory import DEMO_CHAIN, WaveformFactory

OUT = os.path.join(ROOT, "web", "demo", "analysis.json")


def main():
    iq, _ = WaveformFactory(seed=99).generate(**DEMO_CHAIN,
                                              payload_mode="text")
    cfg = Config()
    cfg.modulation.cvnet_enabled = False        # no network, deterministic
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)                            # file_path: no local paths
        try:
            iq.astype(np.complex64).tofile("demo_signal.iq")
            res = Analyzer(cfg, use_cache=False).analyze(
                "demo_signal.iq", sample_rate=1_000_000)
        finally:
            os.chdir(cwd)
    export_json(res, OUT)
    print(f"{OUT}: {res.modulation.prediction}, "
          f"CRC {(res.frames.crc or {}).get('name') if res.frames else None}")


if __name__ == "__main__":
    main()
