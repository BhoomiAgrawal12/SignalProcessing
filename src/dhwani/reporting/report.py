"""Stage S12: machine-readable and analyst-readable exports
(S11 payload intelligence results are included in every format).

JSON / CSV / payload.bin+hex always work; PDF needs reportlab; SigMF
sidecar is plain JSON per the SigMF v1 spec; the .grc export writes a GNU
Radio 3.10 flowgraph pre-configured from the estimated parameters."""
from __future__ import annotations

import csv
import json
import os
from xml.sax.saxutils import escape


def _rows_for_csv(r: dict) -> list:
    rows = [("run_id", r.get("run_id"), "", "")]
    rec = r.get("recording", {})
    for k in ("file_path", "format", "sample_rate", "sample_rate_source",
              "center_frequency", "datatype", "n_samples", "duration_s"):
        rows.append((f"recording.{k}", rec.get(k), "", ""))
    p = r.get("parameters") or {}
    conf = p.get("confidences", {})
    def c(name):
        d = conf.get(name, {})
        return d.get("value", ""), d.get("method", "")
    for key, cname in [("obw99_norm", "obw"), ("carrier_offset_norm", "cfo"),
                       ("snr_db", "snr"), ("symbol_rate_norm", "symbol_rate"),
                       ("symbol_rate_norm_recording", "symbol_rate"),
                       ("samples_per_symbol", "symbol_rate"),
                       ("symbol_rate_hz", "symbol_rate"),
                       ("excess_bandwidth", "obw")]:
        v, m = c(cname)
        rows.append((f"parameters.{key}", p.get(key), v, m))
    m = r.get("modulation") or {}
    rows.append(("modulation.prediction", m.get("prediction"),
                 m.get("confidence"), "fusion of engines"))
    for eng, pred in (m.get("engine_predictions") or {}).items():
        if isinstance(pred, list):
            rows.append((f"modulation.engine.{eng}", pred[0], pred[1], ""))
    s = r.get("scrambler") or {}
    rows.append(("scrambler.kind", s.get("kind"), s.get("score"), s.get("name", "")))
    il = r.get("interleaver") or {}
    rows.append(("interleaver.kind", il.get("label") or il.get("kind"),
                 il.get("score"),
                 json.dumps(il.get("parameters", {}))))
    f = r.get("fec") or {}
    rows.append(("fec.family", f.get("family"), f.get("syndrome_zero_rate"),
                 json.dumps(f.get("parameters", {}))))
    fr = r.get("frames") or {}
    rows.append(("frames.length_bits", fr.get("frame_length_bits"),
                 fr.get("score"), ""))
    rows.append(("frames.sync_word", fr.get("sync_word_hex"), "", ""))
    if fr.get("crc"):
        rows.append(("frames.crc", fr["crc"].get("name"),
                     fr["crc"].get("pass_fraction"), ""))
    pl = r.get("payload") or {}
    rows.append(("payload.n_bytes", pl.get("n_bytes"), "", ""))
    rows.append(("payload.entropy_bits_per_bit", pl.get("entropy_bits_per_bit"),
                 "", "likely encrypted" if pl.get("likely_encrypted") else ""))
    pi = r.get("payload_intelligence") or {}
    if pi.get("available"):
        summ = pi.get("summary", {})
        rows.append(("payload_intel.classification", summ.get("classification"),
                     summ.get("confidence"), summ.get("strength")))
        for f in pi.get("findings", [])[:10]:
            rows.append((f"payload_intel.{f.get('category')}",
                         f.get("verdict"), f.get("confidence"),
                         "; ".join(f.get("evidence", [])[:2])))
        for lim in pi.get("limitations", [])[:4]:
            rows.append(("payload_intel.limitation", lim, "", ""))
    return rows


def export_json(result, path: str):
    with open(path, "w") as f:
        f.write(result.to_json(indent=2) if hasattr(result, "to_json")
                else json.dumps(result, indent=2))
    return path


def _csv_safe(v):
    """Strings opened by a spreadsheet must not run as formulas (OWASP CSV
    injection): prefix the trigger characters. Numbers pass unchanged."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def export_csv(result, path: str):
    r = result.to_dict() if hasattr(result, "to_dict") else result
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "confidence", "method"])
        w.writerows([[_csv_safe(v) for v in row] for row in _rows_for_csv(r)])
    return path


def export_payload(result, path: str):
    payload = getattr(result, "payload", None)
    if payload is None or not payload.data:
        return None
    with open(path, "wb") as f:
        f.write(payload.data)
    hex_path = os.path.splitext(path)[0] + ".hex.txt"
    with open(hex_path, "w") as f:
        data = payload.data
        for i in range(0, len(data), 16):
            chunk = data[i:i + 16]
            hx = " ".join(f"{b:02x}" for b in chunk)
            asc = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in chunk)
            f.write(f"{i:08x}  {hx:<48}  {asc}\n")
    return path


def export_sigmf_meta(result, data_path: str, path: str = None):
    """Write a .sigmf-meta sidecar with the analysis as annotations."""
    r = result.to_dict() if hasattr(result, "to_dict") else result
    rec = r.get("recording", {})
    dt_map = {"complex64": "cf32_le", "int16": "ci16_le", "int8": "ci8",
              "uint8": "cu8", "float32": "cf32_le"}
    meta = {
        "global": {
            "core:datatype": dt_map.get(rec.get("datatype"), "cf32_le"),
            "core:version": "1.0.0",
            "core:description": "Dhwani analysis result",
            **({"core:sample_rate": rec["sample_rate"]}
               if rec.get("sample_rate") else {}),
        },
        "captures": [{"core:sample_start": 0,
                      **({"core:frequency": rec["center_frequency"]}
                         if rec.get("center_frequency") else {})}],
        "annotations": [],
    }
    # segment indices are relative to the analysed window
    offset = int(rec.get("analysed_start") or 0)
    for seg in r.get("segments", []):
        ann = {"core:sample_start": offset + seg.get("start_sample", 0),
               "core:sample_count": max(0, seg.get("end_sample", 0) -
                                        seg.get("start_sample", 0)),
               "core:label": f"signal {seg.get('id')}: "
                             f"SNR {seg.get('snr_db')} dB"}
        if seg.get("f_low_hz") is not None:
            ann["core:freq_lower_edge"] = seg["f_low_hz"]
            ann["core:freq_upper_edge"] = seg["f_high_hz"]
        meta["annotations"].append(ann)
    m = r.get("modulation") or {}
    if m.get("prediction"):
        meta["global"]["dhwani:analysis"] = {
            "modulation": m.get("prediction"),
            "modulation_confidence": m.get("confidence"),
            "fec": (r.get("fec") or {}).get("family"),
            "interleaver": (r.get("interleaver") or {}).get("kind"),
            "frame_length_bits": (r.get("frames") or {}).get("frame_length_bits"),
            "sync_word_hex": (r.get("frames") or {}).get("sync_word_hex"),
        }
    path = path or (os.path.splitext(data_path)[0] + ".sigmf-meta")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def export_pdf(result, path: str):
    """Single-page analyst parameter sheet via reportlab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                        Paragraph, Spacer)
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        return None
    r = result.to_dict() if hasattr(result, "to_dict") else result
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=15 * mm)
    story = [Paragraph("Dhwani - Analysis Report", styles["Title"]),
             Paragraph(escape(f"run {r.get('run_id')} - "
                              f"{r.get('recording', {}).get('file_path', '')}"),
                       styles["Normal"]),
             Spacer(1, 6 * mm)]
    rows = [["Parameter", "Value", "Confidence", "Method"]]
    for row in _rows_for_csv(r):
        rows.append([str(x) if x is not None else "-" for x in row])
    t = Table(rows, colWidths=[55 * mm, 55 * mm, 25 * mm, 45 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f4f8")]),
    ]))
    story.append(t)
    warnings = r.get("warnings", [])
    if warnings:
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph("Warnings", styles["Heading3"]))
        for w in warnings:
            story.append(Paragraph(escape(f"• {w}"), styles["Normal"]))
    pl = r.get("payload") or {}
    if pl.get("hex"):
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph("Payload (first bytes, hex)", styles["Heading3"]))
        story.append(Paragraph(escape(str(pl["hex"])[:512]), styles["Code"]))
    doc.build(story)
    return path


_GRC_TEMPLATE = """options:
  parameters:
    author: dhwani
    catch_exceptions: 'True'
    output_language: python
    generate_options: qt_gui
    id: dhwani_receiver
    title: 'Dhwani generated receiver'
  states:
    coordinate: [8, 8]

blocks:
- name: samp_rate
  id: variable
  parameters: {{value: '{samp_rate}'}}
  states: {{coordinate: [8, 100]}}
- name: sym_rate
  id: variable
  parameters: {{value: '{sym_rate}'}}
  states: {{coordinate: [8, 164]}}
- name: blocks_file_source_0
  id: blocks_file_source
  parameters:
    file: {file_path}
    type: complex
    repeat: 'False'
  states: {{coordinate: [16, 260]}}
- name: freq_xlating_fir_filter_0
  id: freq_xlating_fir_filter_xxx
  parameters:
    center_freq: '{center_freq}'
    decim: '1'
    samp_rate: samp_rate
    taps: firdes.low_pass(1.0, samp_rate, {cutoff}, {trans})
    type: ccc
  states: {{coordinate: [260, 260]}}
{sync_block}- name: qtgui_const_sink_x_0
  id: qtgui_const_sink_x
  parameters:
    name: '"Dhwani constellation"'
    size: '1024'
    type: complex
  states: {{coordinate: [800, 260]}}

connections:
- [blocks_file_source_0, '0', freq_xlating_fir_filter_0, '0']
{sync_connections}
metadata:
  file_format: 1
"""


_GRC_SYNC_BLOCK = """- name: digital_symbol_sync_0
  id: digital_symbol_sync_xx
  parameters:
    constellation: {constellation}
    damping: '1.0'
    loop_bw: '0.045'
    max_dev: '1.5'
    osps: '1'
    resamp_type: digital.IR_MMSE_8TAP
    sps: 'samp_rate/sym_rate'
    ted_gain: '1.0'
    ted_type: digital.TED_SIGNAL_TIMES_SLOPE_ML
    type: cc
  states: {{coordinate: [520, 260]}}
"""

_GRC_CONSTELLATIONS = {
    "BPSK": "digital.constellation_bpsk().base()",
    "QPSK": "digital.constellation_qpsk().base()",
    "8PSK": "digital.constellation_8psk().base()",
    "16QAM": "digital.constellation_16qam().base()",
}


def export_grc(result, path: str):
    """GNU Radio 3.10 flowgraph pre-configured from the analysis.

    Targets GR 3.10 block ids; generated as a convenience starting point
    for an analyst with GNU Radio installed (not executed here). Skipped
    (returns None) when the sample rate is unknown: every block parameter
    is in Hz. The symbol-sync block is emitted only for a modulation with a
    stock GNU Radio constellation object."""
    r = result.to_dict() if hasattr(result, "to_dict") else result
    rec = r.get("recording", {})
    p = r.get("parameters") or {}
    if not rec.get("sample_rate"):
        return None
    # float() on every number: `report` re-renders an untrusted JSON
    samp = float(rec["sample_rate"])
    rs_norm = float(p.get("symbol_rate_norm_recording") or
                    p.get("symbol_rate_norm") or 0.1)
    seg = r.get("selected_segment") or {}
    center = float(seg.get("center_hz") or
                   (float(seg.get("center_norm") or 0.0) * samp))
    bw = float(p.get("obw99_hz") or float(p.get("obw99_norm") or 0.2) * samp)
    mod = str((r.get("modulation") or {}).get("prediction"))
    cst = _GRC_CONSTELLATIONS.get(mod)
    if cst:
        sync_block = _GRC_SYNC_BLOCK.format(constellation=cst)
        sync_connections = (
            "- [freq_xlating_fir_filter_0, '0', digital_symbol_sync_0, '0']\n"
            "- [digital_symbol_sync_0, '0', qtgui_const_sink_x_0, '0']")
    else:
        # json.dumps quotes the untrusted label: no newline survives
        sync_block = (f"# no symbol sync: no stock GNU Radio constellation "
                      f"for {json.dumps(mod)}\n")
        sync_connections = (
            "- [freq_xlating_fir_filter_0, '0', qtgui_const_sink_x_0, '0']")
    content = _GRC_TEMPLATE.format(
        samp_rate=samp, sym_rate=rs_norm * samp,
        # a JSON string is a valid YAML double-quoted scalar: quotes and
        # newlines in an attacker-chosen filename stay inside the value
        file_path=json.dumps(str(rec.get("file_path", ""))),
        center_freq=center, cutoff=bw * 0.75, trans=bw * 0.25,
        sync_block=sync_block, sync_connections=sync_connections)
    with open(path, "w") as f:
        f.write(content)
    return path


def export_all(result, out_dir: str, base_name: str = "analysis") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    b = os.path.join(out_dir, base_name)
    written = {}
    written["json"] = export_json(result, b + ".json")
    written["csv"] = export_csv(result, b + ".csv")
    pl = export_payload(result, b + "_payload.bin")
    if pl:
        written["payload"] = pl
    rec = result.recording_meta if hasattr(result, "recording_meta") else {}
    written["sigmf"] = export_sigmf_meta(result, rec.get("file_path", b),
                                         b + ".sigmf-meta")
    pdf = export_pdf(result, b + ".pdf")
    if pdf:
        written["pdf"] = pdf
    grc = export_grc(result, b + ".grc")
    if grc:
        written["grc"] = grc
    return written
