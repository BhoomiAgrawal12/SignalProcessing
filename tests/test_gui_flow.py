"""R8: drive the GUI as a user would (pytest-qt, offscreen): run an
analysis on a worker thread, check the PS-required views are populated,
then force a modulation override and re-run. Skipped without the [gui]
extra and pytest-qt."""
import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _run(qtbot, win):
    win._run_analysis()
    with qtbot.waitSignal(win.worker.finished_ok, timeout=120_000):
        pass
    qtbot.waitUntil(lambda: win.result is not None and
                    win.btn_analyze.isEnabled(), timeout=10_000)
    return win.result


def test_gui_analysis_flow(qtbot, tmp_path):
    from dhwani.gui.app import MainWindow
    from dhwani.synth.factory import DEMO_CHAIN, WaveformFactory
    iq, _ = WaveformFactory(seed=99).generate(**DEMO_CHAIN)
    p = str(tmp_path / "flow.iq")
    iq.astype(np.complex64).tofile(p)

    win = MainWindow()
    qtbot.addWidget(win)
    win.file_path = p
    win.in_rate.setText("1000000")
    win.in_dtype.setCurrentText("complex64")
    res = _run(qtbot, win)

    # PS GUI requirement: sample rate, waterfall, constellation are shown
    meta = {win.tbl_meta.item(r, 0).text(): win.tbl_meta.item(r, 1).text()
            for r in range(win.tbl_meta.rowCount())}
    assert meta["sample_rate"] == "1000000.0"
    assert win.wf_img.image is not None and win.wf_img.image.size > 0
    # the waterfall x axis is labelled normalised frequency: it must span
    # -0.5..0.5 like the PSD above it, not 0..n_bins
    rect = win.wf_img.mapRectToParent(win.wf_img.boundingRect())
    assert abs(rect.left() + 0.5) < 0.01 and abs(rect.right() - 0.5) < 0.01
    assert win.const_plot.listDataItems() or win.const_plot.getPlotItem().items
    assert res.modulation.prediction == "QPSK"
    assert "CRC" in win.txt_summary.toPlainText()

    # analyst override: force BPSK and re-run; the label says so
    win.cmb_mod.setCurrentText("BPSK")
    win.btn_mod_override.click()
    with qtbot.waitSignal(win.worker.finished_ok, timeout=120_000):
        pass
    qtbot.waitUntil(lambda: win.result.modulation.prediction == "BPSK",
                    timeout=10_000)
    assert "analyst override" in win.result.modulation.constraints_applied


def test_gui_screenshots_script(tmp_path):
    """R8: one PNG per page, non-trivial in size."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import gui_screenshots
    from dhwani.gui.app import PAGES
    from dhwani.synth.factory import DEMO_CHAIN, WaveformFactory
    iq, _ = WaveformFactory(seed=99).generate(**DEMO_CHAIN)
    p = str(tmp_path / "s.iq")
    iq.astype(np.complex64).tofile(p)
    out = gui_screenshots.main([p, "--sample-rate", "1e6", "--datatype",
                                "complex64", "--out", str(tmp_path / "shots")])
    assert len(out) == len(PAGES)
    assert all(os.path.getsize(f) > 5000 for f in out)
