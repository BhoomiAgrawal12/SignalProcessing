"""GUI smoke test: every page renders a real full-chain result offscreen.
Skipped unless the .[gui] extra (PyQt6, pyqtgraph) is installed."""
import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_pages_render_full_result(tmp_path, config):
    from PyQt6.QtWidgets import QApplication
    from dhwani.gui.app import MainWindow
    from dhwani.pipeline import Analyzer
    from dhwani.synth.factory import DEMO_CHAIN, WaveformFactory

    iq, _ = WaveformFactory(seed=99).generate(**DEMO_CHAIN)
    p = str(tmp_path / "g.iq")
    iq.astype(np.complex64).tofile(p)
    res = Analyzer(config, use_cache=False).analyze(p, sample_rate=1e6)

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    assert win._analysis_kwargs()["no_ml"] is win.chk_offline.isChecked()
    win.result = res
    win._refresh_all()
    assert win.tbl_trace.rowCount() == 13            # S0..S12
    assert win.ac_plot.listDataItems()                # 3.E correlation view
    assert "CRC" in win.txt_summary.toPlainText()
    assert win.txt_intel.toPlainText()
    win.close()
    app.processEvents()
