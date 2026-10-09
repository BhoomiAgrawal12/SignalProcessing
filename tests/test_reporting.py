"""Exports must survive hostile file names and warnings (1.C, 1.E)."""
import os

import pytest

from dhwani.reporting.report import export_grc, export_pdf

EVIL = "x'\n- name: pwn\n  id: blocks_null_sink\n"


def _result(**kw):
    r = {"run_id": "t", "recording": {"file_path": EVIL, "sample_rate": 1e6},
         "modulation": {"prediction": "BPSK"}, "warnings": []}
    r.update(kw)
    return r


def test_grc_quotes_hostile_filename(tmp_path):
    path = export_grc(_result(), str(tmp_path / "a.grc"))
    text = open(path).read()
    assert "\n- name: pwn" not in text
    assert "constellation_bpsk" in text
    try:
        import yaml
    except ImportError:
        return
    doc = yaml.safe_load(text)
    names = [b["name"] for b in doc["blocks"]]
    assert "pwn" not in names
    src = next(b for b in doc["blocks"] if b["name"] == "blocks_file_source_0")
    assert src["parameters"]["file"] == EVIL


def test_grc_omits_sync_for_non_psk_and_skips_unknown_rate(tmp_path):
    p = export_grc(_result(modulation={"prediction": "2FSK"}),
                   str(tmp_path / "b.grc"))
    assert "digital_symbol_sync" not in open(p).read()
    assert export_grc(_result(recording={"file_path": "a.iq"}),
                      str(tmp_path / "c.grc")) is None


def test_pdf_escapes_markup(tmp_path):
    pytest.importorskip("reportlab")
    r = _result(recording={"file_path": "/rec/a&b<c.iq"},
                warnings=["level <0 dB & clipped"])
    p = export_pdf(r, str(tmp_path / "a.pdf"))
    assert p and os.path.getsize(p) > 0


def test_sigmf_annotations_include_window_offset(tmp_path):
    """A5: segments are relative to the analysed window; SigMF annotations
    must point into the file."""
    import json
    from dhwani.reporting.report import export_sigmf_meta
    r = {"recording": {"analysed_start": 5_000_000},
         "segments": [{"id": 0, "start_sample": 100, "end_sample": 300}]}
    p = export_sigmf_meta(r, "x.iq", str(tmp_path / "x.sigmf-meta"))
    ann = json.load(open(p))["annotations"][0]
    assert (ann["core:sample_start"], ann["core:sample_count"]) == (5_000_100, 200)


def test_csv_neutralises_formulas(tmp_path):
    """A15: attacker-chosen strings must not become spreadsheet formulas;
    numbers (negative ones included) stay numbers."""
    import csv
    from dhwani.reporting.report import export_csv
    r = {"run_id": "t", "recording": {"file_path": '=HYPERLINK("http://x")',
                                      "format": "@SUM(A1)"},
         "parameters": {"carrier_offset_norm": -0.003}}
    p = export_csv(r, str(tmp_path / "a.csv"))
    rows = {row[0]: row[1] for row in csv.reader(open(p))}
    assert rows["recording.file_path"] == "'=HYPERLINK(\"http://x\")"
    assert rows["recording.format"] == "'@SUM(A1)"
    assert rows["parameters.carrier_offset_norm"] == "-0.003"


def test_grc_symbol_rate_uses_recording_units(tmp_path):
    """sym_rate = (symbols per recorded sample) x sample rate; the
    channelised value would be wrong whenever the channeliser decimates."""
    r = _result(parameters={"symbol_rate_norm": 0.0895,
                            "symbol_rate_norm_recording": 0.025})
    text = open(export_grc(r, str(tmp_path / "s.grc"))).read()
    assert "value: '25000.0'" in text
