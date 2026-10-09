"""Dhwani analyst GUI (PyQt6 + PyQtGraph).

Single window, left rail navigation, all analysis in a worker thread so
the UI never freezes. Uses exactly the same Analyzer engine as the CLI.
Run with:  python -m dhwani.gui.app  (or dhwani-gui)
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget, QLineEdit, QInputDialog)

import pyqtgraph as pg

from ..common.config import load_config
from ..pipeline import Analyzer

pg.setConfigOptions(antialias=False, background="k", foreground="w")

DEMO_WAV = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "web", "demo",
    "demo_chain.wav"))

PAGES = ["Load & Inspect", "Spectrum & Waterfall", "Parameters",
         "Modulation", "Demodulation", "Bit Layer", "Frames & Payload",
         "Report & Export"]


class AnalysisWorker(QThread):
    progressed = pyqtSignal(str, float)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, analyzer, path, kwargs):
        super().__init__()
        self.analyzer = analyzer
        self.path = path
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.analyzer.analyze(
                self.path,
                progress=lambda stage, frac: self.progressed.emit(stage, frac),
                **self.kwargs)
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


def _table(rows, headers):
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            t.setItem(r, c, QTableWidgetItem("" if val is None else str(val)))
    t.resizeColumnsToContents()
    return t


def _fill_table(t: QTableWidget, rows, headers):
    t.clear()
    t.setColumnCount(len(headers))
    t.setRowCount(len(rows))
    t.setHorizontalHeaderLabels(headers)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            t.setItem(r, c, QTableWidgetItem("" if val is None else str(val)))
    t.resizeColumnsToContents()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dhwani - blind SDR signal analysis")
        self.resize(1400, 860)
        self.config = load_config()
        ckpt = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "ml", "cvnet_rf", "checkpoints", "real_best.pt")
        if os.path.exists(ckpt):
            self.config.modulation.cvnet_checkpoint = os.path.abspath(ckpt)
        self.analyzer = Analyzer(self.config)
        self.result = None
        self.worker = None
        self.file_path = None
        self.overrides = {}

        root = QSplitter()
        self.nav = QListWidget()
        self.nav.addItems(PAGES)
        self.nav.setFixedWidth(190)
        self.nav.currentRowChanged.connect(self._switch_page)
        self.stack = QStackedWidget()
        root.addWidget(self.nav)
        root.addWidget(self.stack)
        self.setCentralWidget(root)

        self._build_load_page()
        self._build_spectrum_page()
        self._build_params_page()
        self._build_modulation_page()
        self._build_demod_page()
        self._build_bitlayer_page()
        self._build_frames_page()
        self._build_report_page()
        self.nav.setCurrentRow(0)

        self.status_bar = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.status_bar.addPermanentWidget(self.progress)

    # ------------------------------------------------------------ pages
    def _build_load_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        row = QHBoxLayout()
        self.btn_open = QPushButton("Open recording...")
        self.btn_open.clicked.connect(self._open_file)
        self.btn_demo = QPushButton("Load demo")
        self.btn_demo.clicked.connect(self._load_demo)
        self.btn_demo.setEnabled(os.path.exists(DEMO_WAV))
        self.btn_demo.setToolTip("Synthetic QPSK + FEC recording (web/demo/"
                                 "demo_chain.wav in a repository checkout)")
        self.lbl_file = QLabel("no file loaded")
        row.addWidget(self.btn_open)
        row.addWidget(self.btn_demo)
        row.addWidget(self.lbl_file, 1)
        lay.addLayout(row)

        form_box = QGroupBox("Manual overrides (blank = automatic)")
        form = QFormLayout(form_box)
        self.in_rate = QLineEdit()
        self.in_rate.setPlaceholderText("e.g. 1000000 (Hz); required for "
                                        "absolute units on headerless raw IQ")
        self.in_cf = QLineEdit()
        self.in_cf.setPlaceholderText("centre frequency in Hz")
        self.in_dtype = QComboBox()
        self.in_dtype.addItems(["auto", "complex64", "float32", "int16",
                                "int8", "uint8"])
        form.addRow("Sample rate", self.in_rate)
        form.addRow("Centre frequency", self.in_cf)
        form.addRow("Raw IQ datatype", self.in_dtype)
        # no network unless the analyst opts in: unchecking lets CVNet-RF
        # download its pinned checkpoint when none is present locally
        self.chk_offline = QCheckBox("Offline: skip the CVNet-RF engine")
        self.chk_offline.setChecked(
            not self.config.modulation.cvnet_checkpoint)
        form.addRow("ML engine", self.chk_offline)
        lay.addWidget(form_box)

        self.btn_analyze = QPushButton("Run full analysis")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._run_analysis)
        lay.addWidget(self.btn_analyze)

        self.tbl_meta = QTableWidget()
        lay.addWidget(QLabel("Recording metadata"))
        lay.addWidget(self.tbl_meta, 1)
        self.txt_warnings = QPlainTextEdit()
        self.txt_warnings.setReadOnly(True)
        self.txt_warnings.setMaximumHeight(120)
        lay.addWidget(QLabel("Warnings and quality flags"))
        lay.addWidget(self.txt_warnings)
        self.stack.addWidget(page)

    def _build_spectrum_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.psd_plot = pg.PlotWidget(title="Power spectral density")
        self.psd_plot.setLabel("bottom", "normalised frequency (cycles/sample)")
        self.psd_plot.setLabel("left", "dB")
        lay.addWidget(self.psd_plot, 2)
        self.wf_view = pg.PlotWidget(title="Waterfall (click a box to select the signal)")
        self.wf_img = pg.ImageItem()
        self.wf_view.addItem(self.wf_img)
        self.wf_view.setLabel("bottom", "normalised frequency")
        self.wf_view.setLabel("left", "time block")
        lay.addWidget(self.wf_view, 3)
        self.tbl_signals = QTableWidget()
        self.tbl_signals.cellDoubleClicked.connect(self._select_signal)
        lay.addWidget(QLabel("Detected signals (double-click to re-analyse that signal)"))
        lay.addWidget(self.tbl_signals, 2)
        self.stack.addWidget(page)

    def _build_params_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.tbl_params = QTableWidget()
        lay.addWidget(QLabel("Estimated physical parameters (with confidence and method)"))
        lay.addWidget(self.tbl_params, 1)
        box = QGroupBox("Analyst override")
        form = QFormLayout(box)
        self.in_rs_norm = QLineEdit()
        self.in_rs_norm.setPlaceholderText("symbol rate, symbols/sample "
                                           "(e.g. 0.125)")
        self.btn_rs_override = QPushButton("Pin symbol rate and re-run downstream")
        self.btn_rs_override.clicked.connect(self._override_symbol_rate)
        form.addRow("Symbol rate (normalised)", self.in_rs_norm)
        form.addRow(self.btn_rs_override)
        lay.addWidget(box)
        self.stack.addWidget(page)

    def _build_modulation_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.lbl_mod = QLabel("-")
        f = QFont(); f.setPointSize(18); f.setBold(True)
        self.lbl_mod.setFont(f)
        lay.addWidget(self.lbl_mod)
        self.lbl_mod_note = QLabel("")
        self.lbl_mod_note.setWordWrap(True)
        lay.addWidget(self.lbl_mod_note)
        self.tbl_mod = QTableWidget()
        lay.addWidget(QLabel("Ranked candidates and per-engine predictions"))
        lay.addWidget(self.tbl_mod, 1)
        box = QGroupBox("Analyst override")
        h = QHBoxLayout(box)
        self.cmb_mod = QComboBox()
        self.cmb_mod.addItems(["BPSK", "QPSK", "8PSK", "16QAM", "64QAM",
                               "2FSK", "4FSK"])
        self.btn_mod_override = QPushButton("Force modulation and re-run downstream")
        self.btn_mod_override.clicked.connect(self._override_modulation)
        h.addWidget(self.cmb_mod)
        h.addWidget(self.btn_mod_override)
        lay.addWidget(box)
        self.stack.addWidget(page)

    def _build_demod_page(self):
        page = QWidget()
        lay = QHBoxLayout(page)
        left = QVBoxLayout()
        self.const_plot = pg.PlotWidget(title="Constellation")
        self.const_plot.setAspectLocked(True)
        left.addWidget(self.const_plot)
        self.eye_plot = pg.PlotWidget(title="Eye diagram (real part)")
        left.addWidget(self.eye_plot)
        lay.addLayout(left, 3)
        right = QVBoxLayout()
        self.tbl_demod = QTableWidget()
        right.addWidget(QLabel("Synchronisation metrics"))
        right.addWidget(self.tbl_demod)
        lay.addLayout(right, 2)
        self.stack.addWidget(page)

    def _build_bitlayer_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.rank_plot = pg.PlotWidget(
            title="GF(2) rank-deficiency profile (dips mark hidden linear structure)")
        self.rank_plot.setLabel("bottom", "trial row length L (bits)")
        self.rank_plot.setLabel("left", "deficiency significance")
        lay.addWidget(self.rank_plot, 2)
        self.tbl_bitlayer = QTableWidget()
        lay.addWidget(QLabel("Scrambler / interleaver / FEC verdicts"))
        lay.addWidget(self.tbl_bitlayer, 1)
        self.tbl_hyps = QTableWidget()
        lay.addWidget(QLabel("Hypothesis search (all candidates with evidence)"))
        lay.addWidget(self.tbl_hyps, 2)
        self.stack.addWidget(page)

    def _build_frames_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.entropy_plot = pg.PlotWidget(
            title="Per-column entropy field map (0 = sync/constant, 1 = payload/random)")
        self.entropy_plot.setLabel("bottom", "bit position in frame")
        self.entropy_plot.setLabel("left", "entropy (bits)")
        lay.addWidget(self.entropy_plot, 2)
        self.ac_plot = pg.PlotWidget(
            title="Bit-stream autocorrelation (peaks mark the frame period)")
        self.ac_plot.setLabel("bottom", "lag (bits)")
        self.ac_plot.setLabel("left", "fraction of equal bits")
        lay.addWidget(self.ac_plot, 2)
        self.tbl_frames = QTableWidget()
        lay.addWidget(QLabel("Frame structure"))
        lay.addWidget(self.tbl_frames, 1)
        self.txt_hex = QPlainTextEdit()
        self.txt_hex.setReadOnly(True)
        self.txt_hex.setFont(QFont("Menlo", 10))
        lay.addWidget(QLabel("Payload (hex / ASCII)"))
        lay.addWidget(self.txt_hex, 2)
        self.txt_intel = QPlainTextEdit()
        self.txt_intel.setReadOnly(True)
        lay.addWidget(QLabel("S11 payload intelligence (findings capped by "
                             "provenance)"))
        lay.addWidget(self.txt_intel, 2)
        self.stack.addWidget(page)

    def _build_report_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.btn_export = QPushButton("Export all (JSON, CSV, payload, SigMF, PDF, GRC)...")
        self.btn_export.clicked.connect(self._export_all)
        self.btn_export.setEnabled(False)
        lay.addWidget(self.btn_export)
        self.btn_signature = QPushButton("Save to signature library...")
        self.btn_signature.clicked.connect(self._save_signature)
        self.btn_signature.setEnabled(False)
        lay.addWidget(self.btn_signature)
        self.txt_summary = QPlainTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setFont(QFont("Menlo", 11))
        lay.addWidget(QLabel("Analysis summary"))
        lay.addWidget(self.txt_summary, 1)
        self.tbl_trace = QTableWidget()
        lay.addWidget(QLabel("Pipeline trace (S0-S12)"))
        lay.addWidget(self.tbl_trace, 1)
        self.stack.addWidget(page)

    # --------------------------------------------------------- actions
    def _switch_page(self, idx):
        self.stack.setCurrentIndex(idx)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", "",
            "Recordings (*.iq *.wav *.sigmf-data *.bin *.raw);;All files (*)")
        if path:
            self._set_file(path)

    def _load_demo(self):
        self._set_file(DEMO_WAV)

    def _set_file(self, path):
        self.file_path = path
        self.lbl_file.setText(path)
        self.btn_analyze.setEnabled(True)
        self.overrides = {}

    def _analysis_kwargs(self):
        kw = {"overrides": dict(self.overrides),
              "no_ml": self.chk_offline.isChecked()}
        if self.in_rate.text().strip():
            kw["sample_rate"] = float(self.in_rate.text())
        if self.in_cf.text().strip():
            kw["center_frequency"] = float(self.in_cf.text())
        if self.in_dtype.currentText() != "auto":
            kw["datatype"] = self.in_dtype.currentText()
        return kw

    def _run_analysis(self):
        if not self.file_path or (self.worker and self.worker.isRunning()):
            return
        self.btn_analyze.setEnabled(False)
        self.status_bar.showMessage("analysis running...")
        self.progress.setValue(0)
        self.worker = AnalysisWorker(self.analyzer, self.file_path,
                                     self._analysis_kwargs())
        self.worker.progressed.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_result)
        self.worker.failed.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, stage, frac):
        stages = ["ingest", "condition", "detect", "channelize", "parameters",
                  "modulation", "demodulate", "ambiguity", "bitlayer"]
        base = stages.index(stage) if stage in stages else 0
        self.progress.setValue(int(100 * (base + frac) / len(stages)))
        msg = f"stage: {stage}"
        if stage == "bitlayer":     # reports once per candidate, can be slow
            msg += (f" ({int(100 * frac)}% of candidate streams tried; "
                    "this stage can take a minute or two)")
        self.status_bar.showMessage(msg)

    def _on_error(self, tb):
        self.btn_analyze.setEnabled(True)
        self.status_bar.showMessage("analysis failed")
        QMessageBox.critical(self, "Analysis failed", tb[-2000:])

    def _on_result(self, result):
        self.result = result
        self.btn_analyze.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_signature.setEnabled(True)
        self.progress.setValue(100)
        self.status_bar.showMessage(
            f"analysis complete (run {result.run_id})")
        self._refresh_all()

    def _select_signal(self, row, col):
        self.overrides["signal_index"] = row
        self._run_analysis()

    def _override_modulation(self):
        self.overrides["modulation"] = self.cmb_mod.currentText()
        self._run_analysis()

    def _override_symbol_rate(self):
        try:
            self.overrides["symbol_rate_norm"] = float(self.in_rs_norm.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid value",
                                "Enter the symbol rate in symbols/sample, "
                                "e.g. 0.125")
            return
        self._run_analysis()

    def _export_all(self):
        out = QFileDialog.getExistingDirectory(self, "Export directory")
        if not out:
            return
        from ..reporting import export_all
        written = export_all(self.result, out)
        QMessageBox.information(self, "Exported",
                                "\n".join(f"{k}: {v}" for k, v in written.items()))

    def _save_signature(self):
        name, ok = QInputDialog.getText(self, "Signature name",
                                        "Name for this waveform signature:")
        if not ok or not name:
            return
        from ..signatures import SignatureDB
        db = SignatureDB(self.config.signature_db)
        sid = db.save_from_result(self.result, name)
        QMessageBox.information(self, "Saved", f"signature id {sid}")

    # --------------------------------------------------------- refresh
    def _refresh_all(self):
        r = self.result
        d = r.to_dict()
        rec = d["recording"]
        _fill_table(self.tbl_meta,
                    [[k, v] for k, v in rec.items()
                     if k not in ("sniff_report", "warnings")],
                    ["field", "value"])
        self.txt_warnings.setPlainText("\n".join(d.get("warnings", [])) or
                                       "none")

        plots = d.get("plots", {})
        # PSD
        self.psd_plot.clear()
        psd = plots.get("psd")
        if psd:
            self.psd_plot.plot(psd["freq_norm"], psd["psd_db"],
                               pen=pg.mkPen("#4fc3f7", width=1))
        # waterfall
        wf = plots.get("waterfall")
        if wf and wf.get("db"):
            img = np.array(wf["db"], dtype=np.float32)
            self.wf_img.setImage(img.T, autoLevels=True)
            # map pixels onto the axis labels: x in normalised frequency
            # (same span as the PSD), y in time blocks
            f = wf.get("freq_norm") or [-0.5, 0.5]
            self.wf_img.setRect(f[0], 0, f[-1] - f[0], img.shape[0])
        # signals table
        segs = d.get("segments", [])
        _fill_table(self.tbl_signals,
                    [[s.get("id"), f"{s.get('center_norm'):+.4f}",
                      f"{s.get('bandwidth_norm'):.4f}",
                      s.get("center_hz"), s.get("bandwidth_hz"),
                      f"{s.get('snr_db'):.1f}"] for s in segs],
                    ["id", "centre (norm)", "bw (norm)", "centre Hz",
                     "bw Hz", "SNR dB"])
        # parameters
        p = d.get("parameters") or {}
        conf = p.get("confidences", {})
        rows = []
        for key in ("obw99_norm", "obw3db_norm", "carrier_offset_norm",
                    "snr_db", "symbol_rate_norm", "samples_per_symbol",
                    "symbol_rate_hz", "carrier_offset_hz", "obw99_hz",
                    "excess_bandwidth", "fsk_tone_count",
                    "fsk_deviation_norm", "ofdm_detected"):
            cname = {"obw99_norm": "obw", "obw3db_norm": "obw",
                     "obw99_hz": "obw", "carrier_offset_norm": "cfo",
                     "carrier_offset_hz": "cfo", "snr_db": "snr",
                     "symbol_rate_norm": "symbol_rate",
                     "symbol_rate_hz": "symbol_rate",
                     "samples_per_symbol": "symbol_rate",
                     "fsk_tone_count": "fsk",
                     "fsk_deviation_norm": "fsk",
                     "ofdm_detected": "ofdm"}.get(key, "")
            cd = conf.get(cname, {})
            rows.append([key, p.get(key), cd.get("value"), cd.get("method")])
        _fill_table(self.tbl_params, rows,
                    ["parameter", "value", "confidence", "method"])
        # modulation
        m = d.get("modulation") or {}
        self.lbl_mod.setText(f"{m.get('prediction', '-')}   "
                             f"confidence {m.get('confidence', 0)}")
        notes = []
        if not m.get("classifier_agreement", True):
            notes.append("WARNING: engines disagree - inspect before trusting")
        if not m.get("in_distribution", True):
            notes.append("signal may be outside the classifier training set")
        notes.extend(m.get("constraints_applied", []))
        self.lbl_mod_note.setText(" | ".join(str(n) for n in notes))
        rows = [["fusion: " + str(l), f"{pr:.3f}", ""]
                for l, pr in m.get("alternatives", [])]
        for eng, pred in (m.get("engine_predictions") or {}).items():
            if isinstance(pred, list):
                rows.append([f"engine {eng}", pred[0], pred[1]])
            elif isinstance(pred, dict):
                rows.append([f"engine {eng}",
                             ", ".join(f"{k}={v}" for k, v in pred.items()), ""])
        _fill_table(self.tbl_mod, rows, ["candidate/engine", "value", "score"])
        # demod
        self.const_plot.clear()
        c = plots.get("constellation")
        if c:
            sp = pg.ScatterPlotItem(x=c["i"], y=c["q"], size=3,
                                    brush=pg.mkBrush(79, 195, 247, 120),
                                    pen=None)
            self.const_plot.addItem(sp)
        self.eye_plot.clear()
        eye = plots.get("eye")
        if eye:
            for tr in eye["traces"][:80]:
                self.eye_plot.plot(tr, pen=pg.mkPen(79, 195, 247, 60))
        dm = d.get("demodulation") or {}
        _fill_table(self.tbl_demod,
                    [[k, v] for k, v in dm.items() if k != "lock_metrics"] +
                    [[f"lock.{k}", v] for k, v in
                     (dm.get("lock_metrics") or {}).items()],
                    ["metric", "value"])
        # bit layer
        self.rank_plot.clear()
        rp = plots.get("rank_profile")
        if rp and rp.get("L"):
            self.rank_plot.plot(rp["L"], rp["significance"],
                                pen=pg.mkPen("#ffb74d", width=2))
        rows = []
        s = d.get("scrambler")
        if s:
            rows.append(["scrambler", s.get("kind"),
                         f"{s.get('name', '')} phase={s.get('seed')}",
                         s.get("score")])
        il = d.get("interleaver")
        if il:
            rows.append(["interleaver", il.get("label") or il.get("kind"),
                         str(il.get("parameters")), il.get("score")])
        f = d.get("fec")
        if f:
            rows.append(["FEC", f.get("family"), str(f.get("parameters")),
                         f"szr={f.get('syndrome_zero_rate')}"])
        _fill_table(self.tbl_bitlayer, rows,
                    ["stage", "verdict", "parameters", "evidence"])
        hyps = d.get("hypotheses", [])
        _fill_table(self.tbl_hyps,
                    [[h.get("stage"), h.get("status"), h.get("score"),
                      str(h.get("assumptions"))[:160]] for h in hyps],
                    ["stage", "status", "score", "assumptions"])
        # frames
        self.entropy_plot.clear()
        fr = d.get("frames") or {}
        ent = fr.get("column_entropy") or plots.get("entropy_map")
        if ent:
            x = list(range(len(ent)))
            bg = pg.BarGraphItem(x=x, height=ent, width=0.9,
                                 brush=pg.mkBrush("#4fc3f7"))
            self.entropy_plot.addItem(bg)
        rows = [["frame length (bits)", fr.get("frame_length_bits")],
                ["sync word", fr.get("sync_word_hex")],
                ["sync offset", fr.get("sync_offset")],
                ["CRC", str(fr.get("crc"))]]
        for fld in fr.get("field_map", []):
            rows.append([f"field @{fld['start_bit']}",
                         f"{fld['role']} ({fld['length_bits']} bits, "
                         f"H={fld['mean_entropy']})"])
        _fill_table(self.tbl_frames, rows, ["field", "value"])
        pl = d.get("payload") or {}
        if pl.get("hex"):
            data = bytes.fromhex(pl["hex"])
            lines = []
            for i in range(0, min(len(data), 2048), 16):
                chunk = data[i:i + 16]
                hx = " ".join(f"{b:02x}" for b in chunk)
                asc = "".join(chr(b) if 0x20 <= b < 0x7F else "."
                              for b in chunk)
                lines.append(f"{i:08x}  {hx:<48}  {asc}")
            extra = (f"\nentropy {pl.get('entropy_bits_per_bit')} bits/bit"
                     + ("  (LIKELY ENCRYPTED - no recovery attempted)"
                        if pl.get("likely_encrypted") else ""))
            self.txt_hex.setPlainText("\n".join(lines) + extra)
        else:
            self.txt_hex.setPlainText("no payload recovered")
        self.ac_plot.clear()
        ac = plots.get("bit_autocorr")
        if ac:
            self.ac_plot.plot(list(range(1, len(ac))), ac[1:],
                              pen=pg.mkPen("#4fc3f7", width=1))
        self.txt_intel.setPlainText(self._intel_text(
            d.get("payload_intelligence") or {}))
        _fill_table(self.tbl_trace,
                    [[t.get("stage"), t.get("title"), t.get("status"),
                      t.get("summary"), t.get("elapsed_s")]
                     for t in d.get("pipeline_trace") or []],
                    ["stage", "title", "status", "summary", "s"])
        # summary
        self.txt_summary.setPlainText(self._summary_text(d))

    @staticmethod
    def _intel_text(pi):
        if not pi.get("available"):
            return pi.get("reason") or "not available"
        s = pi.get("summary") or {}
        lines = [f"{s.get('classification')} ({s.get('strength')}, "
                 f"confidence {s.get('confidence')}); trust {s.get('trust')}",
                 ""]
        for f in pi.get("findings", []):
            lines.append(f"[{f.get('strength')}] {f.get('category')}: "
                         f"{f.get('verdict')} ({f.get('confidence')})")
            lines.extend(f"    - {e}" for e in f.get("evidence", [])[:3])
        lines.extend(f"limitation: {l}" for l in pi.get("limitations", []))
        return "\n".join(lines)

    def _summary_text(self, d):
        lines = [f"run {d.get('run_id')}",
                 f"file {d.get('recording', {}).get('file_path')}", ""]
        m = d.get("modulation") or {}
        p = d.get("parameters") or {}
        lines.append(f"modulation: {m.get('prediction')} "
                     f"(confidence {m.get('confidence')})")
        lines.append(f"SNR: {p.get('snr_db')} dB | symbol rate: "
                     f"{p.get('symbol_rate_norm_recording') or p.get('symbol_rate_norm')} "
                     "symbols/recorded sample"
                     + (f" = {p.get('symbol_rate_hz'):.0f} Bd"
                        if p.get("symbol_rate_hz") else ""))
        s = d.get("scrambler") or {}
        lines.append(f"scrambler: {s.get('kind')} {s.get('name', '')}")
        il = d.get("interleaver") or {}
        lines.append(f"interleaver: {il.get('label') or il.get('kind')} "
                     f"{il.get('parameters')}")
        f = d.get("fec") or {}
        lines.append(f"FEC: {f.get('family')} {f.get('parameters')} "
                     f"syndrome-zero-rate {f.get('syndrome_zero_rate')}")
        fr = d.get("frames") or {}
        lines.append(f"frames: {fr.get('frame_length_bits')} bits, "
                     f"sync {fr.get('sync_word_hex')}, "
                     f"CRC {(fr.get('crc') or {}).get('name')}")
        pl = d.get("payload") or {}
        lines.append(f"payload: {pl.get('n_bytes')} bytes, entropy "
                     f"{pl.get('entropy_bits_per_bit')} bits/bit")
        lines.append("")
        for w in d.get("warnings", []):
            lines.append(f"warning: {w}")
        for t, secs in (d.get("stage_timings") or {}).items():
            lines.append(f"  {t}: {secs}s")
        return "\n".join(lines)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
