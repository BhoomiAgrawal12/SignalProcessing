"""Provenance block written into every generated results/*.json, so each
number in the README can be traced to a machine, a commit and a command."""
import json
import os
import platform
import subprocess
import sys
import time

import numpy
import scipy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu():
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def provenance(data: str = "synthetic (dhwani.synth.WaveformFactory)") -> dict:
    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        # results/ itself is excluded: each script rewrites it, which would
        # otherwise mark every later run of the same batch dirty
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no",
                               "--", ".", ":(exclude)results")),
        "cpu": _cpu(), "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "data": data,
    }


def write_result(name: str, payload: dict, data: str = None) -> str:
    """results/<name>.json = {"provenance": ..., **payload}."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name + ".json")
    prov = provenance(data) if data else provenance()
    with open(path, "w") as f:
        json.dump({"provenance": prov, **payload}, f, indent=2, default=str)
        f.write("\n")
    return path
