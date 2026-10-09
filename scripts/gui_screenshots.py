"""Screenshot every GUI page after analysing one recording (offscreen Qt).

Run:  python scripts/gui_screenshots.py RECORDING [--sample-rate HZ]
                                        [--datatype complex64] [--out DIR]
Writes DIR/<nn>_<page>.png (default results/screens/) for the README and
the demo GIF. Use a real recording: these images are evidence that the
PS-required views (sample rate, waterfall, constellation) work on it.
"""
import argparse
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--sample-rate", type=float, default=None)
    ap.add_argument("--datatype", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "screens"))
    args = ap.parse_args(argv)

    from PyQt6.QtWidgets import QApplication
    from dhwani.gui.app import PAGES, MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.resize(1400, 860)
    win.file_path = args.recording
    win.result = win.analyzer.analyze(
        args.recording, sample_rate=args.sample_rate, datatype=args.datatype,
        no_ml=win.chk_offline.isChecked())
    win._refresh_all()
    os.makedirs(args.out, exist_ok=True)
    written = []
    for i, page in enumerate(PAGES):
        win.nav.setCurrentRow(i)
        app.processEvents()
        name = page.lower().replace(" & ", "_").replace(" ", "_")
        path = os.path.join(args.out, f"{i:02d}_{name}.png")
        if not win.grab().save(path):
            raise RuntimeError(f"could not save {path}")
        written.append(path)
    print("\n".join(written))
    return written


if __name__ == "__main__":
    main()
