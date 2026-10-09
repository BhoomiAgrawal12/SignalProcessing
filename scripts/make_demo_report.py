"""Regenerate the shared demo: web/demo/demo_chain.wav (stereo 16-bit I/Q,
1 MS/s in the header; the "demo" button in the GUI and in web/analyze.html) and
web/demo/analysis.json (the viewer's ?demo=1 report), which is the analysis
of that same file. Seeded synthetic recording: the examples/demo.py chain
with a text payload, so the S11 stage has something to show.

Run:  python scripts/make_demo_report.py
"""
import logging
import os
import shutil
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
logging.disable(logging.INFO)

from dhwani.common.config import Config
from dhwani.pipeline import Analyzer
from dhwani.reporting.report import export_json
from dhwani.synth.factory import DEMO_CHAIN, WaveformFactory
from dhwani.synth.writers import write_wav

OUT = os.path.join(ROOT, "web", "demo", "analysis.json")
WAV = os.path.join(ROOT, "web", "demo", "demo_chain.wav")
SAMPLE_RATE = 1_000_000


def make_wav(path=WAV):
    iq, gt = WaveformFactory(seed=99).generate(**DEMO_CHAIN,
                                               payload_mode="text")
    write_wav(iq, gt, path, SAMPLE_RATE, bits=16)
    os.remove(os.path.splitext(path)[0] + "_truth.json")   # not shipped


def main():
    make_wav()
    cfg = Config()
    cfg.modulation.cvnet_enabled = False        # no network, deterministic
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)                            # file_path: no local paths
        try:
            shutil.copy(WAV, "demo_chain.wav")
            res = Analyzer(cfg, use_cache=False).analyze("demo_chain.wav")
        finally:
            os.chdir(cwd)
    export_json(res, OUT)
    print(f"{OUT}: {res.modulation.prediction}, "
          f"CRC {(res.frames.crc or {}).get('name') if res.frames else None}")


if __name__ == "__main__":
    main()
