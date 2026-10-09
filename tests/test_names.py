"""2.G: the project is Dhwani; no legacy name survives outside CLAUDE.md."""
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
# built by concatenation so this file does not match itself
LEGACY = re.compile("rf" + r"[-_ ]?analy[sz]er", re.I)
SKIP_DIRS = {".git", "__pycache__", ".venv", ".pytest_cache", "node_modules"}


def test_no_legacy_names():
    hits = []
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for name in files:
            if name == "CLAUDE.md":
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            hits += [f"{os.path.relpath(path, ROOT)}: {m.group()}"
                     for m in LEGACY.finditer(text)]
    assert not hits, hits[:10]
