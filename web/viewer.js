/* Report viewer: renders a full desktop analysis.json into the tabbed
 * console. Gracefully handles reports from older versions (missing
 * payload_intelligence or pipeline_trace). */
"use strict";

const $ = UI.$;
let REPORT = null;

/* ------------- load ------------- */
(function setupDrop() {
  const drop = $("drop");
  const input = document.createElement("input");
  input.type = "file"; input.accept = ".json"; input.style.display = "none";
  drop.appendChild(input);
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files.length && read(input.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("hover"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("hover");
    if (e.dataTransfer.files.length) read(e.dataTransfer.files[0]);
  });
  function read(f) {
    if (!(f.name || "").toLowerCase().endsWith(".json")) {
      if (confirm("This page views exported analysis.json reports. To " +
          "analyse a raw .wav/.iq recording in the browser, use the " +
          "Analyse page. Go there now?")) location.href = "analyze.html";
      return;
    }
    const r = new FileReader();
    r.onload = () => {
      try { render(JSON.parse(r.result)); }
      catch (err) { alert("Not a valid analysis JSON: " + err); }
    };
    r.readAsText(f);
  }
  if (new URLSearchParams(location.search).get("demo"))
    fetch("demo/analysis.json").then((r) => r.json()).then(render)
      .catch((e) => alert("demo report unavailable: " + e));
})();

const tabs = UI.initTabs("tabs", (name) => {
  /* charts need a layout pass after their pane becomes visible */
  requestAnimationFrame(() => drawCharts(name));
});

/* ------------- render ------------- */
function fmtHz(v) {
  if (v == null) return null;
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(3) + " MHz";
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(2) + " kHz";
  return v.toFixed(1) + " Hz";
}

function render(d) {
  REPORT = d;
  $("drop").style.display = "none";
  $("report").style.display = "block";
  const rec = d.recording || {};
  $("repline").innerHTML =
    `run <b>${UI.esc(d.run_id || "?")}</b> &middot; ` +
    `${UI.esc((rec.file_path || "").split("/").pop())} &middot; ` +
    `${rec.format || "?"} &middot; ` +
    `${rec.sample_rate ? fmtHz(rec.sample_rate) + " sample rate" : "sample rate unknown"}` +
    `<span class="cursor"></span>`;

  renderOverview(d);
  renderPipelineTab(d);
  renderSpectrumTab(d);
  renderDemodTab(d);
  renderBitlayerTab(d);
  renderFramesTab(d);
  UI.renderPayloadIntel("intel_root", d.payload_intelligence,
                        (d.payload || {}).hex);
  renderLogTab(d);
  tabs.show("overview");
  requestAnimationFrame(() => drawCharts("overview"));
}

function renderOverview(d) {
  const m = d.modulation || {}, p = d.parameters || {}, fec = d.fec || {};
  const fr = d.frames || {}, pl = d.payload || {};
  const pi = d.payload_intelligence || {};
  const crc = fr.crc;
  const conf = m.confidence || 0;
  $("ov_tiles").innerHTML =
    UI.tile("modulation", UI.esc(m.prediction || "unknown"),
            UI.meter(conf) + (m.classifier_agreement === false ?
              " " + UI.badge("ENGINES DISAGREE", "warn") : ""), true) +
    UI.tile("symbol rate", p.symbol_rate_hz ?
            `${(p.symbol_rate_hz / 1e3).toFixed(2)} <small>kBd</small>` :
            `${p.symbol_rate_norm ?? "-"} <small>norm</small>`,
            p.samples_per_symbol ? `${p.samples_per_symbol} samples/symbol` : "") +
    UI.tile("SNR", `${p.snr_db ?? "-"} <small>dB</small>`,
            ((d.parameters || {}).confidences || {}).snr ?
            UI.esc(d.parameters.confidences.snr.method) : "") +
    UI.tile("FEC", UI.esc(fec.family || "-"),
            fec.syndrome_zero_rate != null ?
            `syndrome-zero ${(fec.syndrome_zero_rate * 100).toFixed(1)}%` : "") +
    UI.tile("frame / CRC",
            fr.frame_length_bits ? `${fr.frame_length_bits} <small>bits</small>` : "-",
            crc ? UI.badge(`${crc.name} ${Math.round((crc.pass_fraction || 0) * 100)}%`,
                           crc.pass_fraction > 0.9 ? "good" : "warn")
                : UI.badge("no CRC found", "dim")) +
    UI.tile("payload", pl.n_bytes != null ? `${pl.n_bytes} <small>bytes</small>` : "-",
            pi.summary ? `${UI.esc(pi.summary.classification)} ` +
              UI.strengthBadge(pi.summary.strength) :
            (pl.likely_encrypted ? "high entropy" : ""));

  const rec = d.recording || {};
  UI.kvTable("ov_identity", [
    ["file", UI.esc(rec.file_path || "-")],
    ["format", `${rec.format || "-"} / ${rec.datatype || "-"}`],
    ["sample rate", rec.sample_rate ? `${fmtHz(rec.sample_rate)} ` +
      `<span class="mono">(${rec.sample_rate_source})</span>` :
      "unknown (headerless raw IQ)"],
    ["centre frequency", rec.center_frequency ?
      fmtHz(rec.center_frequency) : "-"],
    ["duration", rec.duration_s ? rec.duration_s.toFixed(4) + " s" : "-"],
    ["carrier offset", p.carrier_offset_norm != null ?
      `${p.carrier_offset_norm} norm` +
      (p.carrier_offset_hz ? ` = ${fmtHz(p.carrier_offset_hz)}` : "") : "-"],
    ["occupied bandwidth", p.obw99_norm != null ?
      `${p.obw99_norm} norm` + (p.obw99_hz ? ` = ${fmtHz(p.obw99_hz)}` : "") : "-"],
    ["excess bandwidth", p.excess_bandwidth ?? "-"],
  ]);

  const scr = d.scrambler || {}, il = d.interleaver || {};
  UI.kvTable("ov_bitlayer", [
    ["scrambler", `${UI.esc(scr.name || scr.kind || "-")}` +
      (scr.seed != null ? ` <span class="mono">phase ${scr.seed}</span>` : "")],
    ["interleaver", `${UI.esc(il.kind || "-")} ` +
      `<span class="mono">${UI.esc(JSON.stringify(il.parameters || {}))}</span>`],
    ["FEC", `${UI.esc(fec.family || "-")} ` +
      `<span class="mono">${UI.esc(JSON.stringify(fec.parameters || {}))}</span>`],
    ["sync word", fr.sync_word_hex ?
      `<span class="mono">${UI.esc(fr.sync_word_hex)}</span>` : "-"],
    ["CRC", crc ? `${crc.name} &middot; passes ${crc.passes}/${crc.total}` : "none found"],
    ["hypotheses examined", (d.hypotheses || []).length],
  ]);
  $("ov_warn").innerHTML = (d.warnings || []).slice(0, 4)
    .map((w) => UI.badge("!", "warn") + ` <span style="color:var(--ink2);` +
         `font-size:0.82rem">${UI.esc(w)}</span>`).join("<br>");
}

function renderPipelineTab(d) {
  let trace = d.pipeline_trace;
  if (!trace || !trace.length) {
    /* older reports: synthesize from stage timings */
    trace = Object.entries(d.stage_timings || {}).map(([k, v]) => ({
      stage: k.split("_")[0], title: k.replace(/_/g, " "),
      status: "executed", summary: "", elapsed_s: v }));
    trace.push({ stage: "-", title: "pipeline trace not recorded by this " +
      "analyzer version", status: "skipped", summary: "" });
  }
  UI.renderPipeline("flow", trace, { detail: (sid) => stageDetail(sid, d) });
}

function stageDetail(sid, d) {
  const t = UI.el("table", "kv");
  const rowsFor = {
    S0: () => Object.entries(d.recording || {})
      .filter(([k]) => !["sniff_report", "warnings"].includes(k))
      .map(([k, v]) => [k, UI.esc(String(v))]),
    S1: () => Object.entries(d.conditioning || {})
      .filter(([k]) => k !== "warnings")
      .map(([k, v]) => [k, UI.esc(JSON.stringify(v))]),
    S2: () => (d.segments || []).map((s) => [`signal ${s.id}`,
      `centre ${(+s.center_norm).toFixed(4)}, bw ${(+s.bandwidth_norm).toFixed(4)}, ` +
      `SNR ${s.snr_db} dB`]),
    S3: () => Object.entries(d.selected_segment || {})
      .map(([k, v]) => [k, UI.esc(String(v))]),
    S4: () => Object.entries(d.parameters || {})
      .filter(([k, v]) => k !== "confidences" && v !== null)
      .map(([k, v]) => [k, UI.esc(JSON.stringify(v))]),
    S5: () => {
      const m = d.modulation || {};
      return [["prediction", `${m.prediction} (${m.confidence})`],
              ...(m.alternatives || []).map((a) => ["candidate", `${a[0]}: ${a[1]}`]),
              ...Object.entries(m.engine_predictions || {}).map(
                ([k, v]) => ["engine " + k, UI.esc(JSON.stringify(v))])];
    },
    S6: () => Object.entries(d.demodulation || {})
      .map(([k, v]) => [k, UI.esc(JSON.stringify(v))]),
    S7: () => [["scrambler", UI.esc(JSON.stringify(d.scrambler || {}))],
               ["candidate streams", (d.hypotheses || []).length]],
    S8: () => [["verdict", UI.esc(JSON.stringify(
      { kind: (d.interleaver || {}).kind,
        parameters: (d.interleaver || {}).parameters }))]],
    S9: () => [["verdict", UI.esc(JSON.stringify(
      { family: (d.fec || {}).family, parameters: (d.fec || {}).parameters,
        syndrome_zero_rate: (d.fec || {}).syndrome_zero_rate }))]],
    S10: () => {
      const fr = d.frames || {};
      return [["frame bits", fr.frame_length_bits],
              ["sync", fr.sync_word_hex],
              ["crc", UI.esc(JSON.stringify(fr.crc))]];
    },
    S11: () => {
      const pi = d.payload_intelligence || {};
      if (!pi.available) return [["status", pi.reason || "not available"]];
      return [["classification",
               `${pi.summary.classification} (${pi.summary.strength})`],
              ...(pi.findings || []).slice(0, 6).map((f) =>
                [f.category, `${UI.esc(f.verdict)} @ ${f.confidence}`])];
    },
    S12: () => [["formats", "JSON, CSV, payload.bin + hex, SigMF sidecar, " +
                 "PDF sheet, GNU Radio flowgraph"]],
  }[sid];
  const rows = rowsFor ? rowsFor() : [];
  if (!rows.length) return null;
  UI.kvTable(t, rows.slice(0, 14));
  return t;
}

function renderSpectrumTab(d) {
  UI.kvTable("signals", (d.segments || []).map((s) => [
    s.id, (+s.center_norm).toFixed(4), (+s.bandwidth_norm).toFixed(4),
    s.center_hz ? fmtHz(s.center_hz) : "-",
    s.bandwidth_hz ? fmtHz(s.bandwidth_hz) : "-",
    `${s.snr_db} dB`]),
    ["id", "centre (norm)", "bw (norm)", "centre", "bandwidth", "SNR"]);
}

function renderDemodTab(d) {
  const dm = d.demodulation || {};
  UI.kvTable("demod_tbl",
    [...Object.entries(dm).filter(([k]) => k !== "lock_metrics"),
     ...Object.entries(dm.lock_metrics || {}).map(([k, v]) => ["lock." + k, v])]
      .map(([k, v]) => [k, UI.esc(JSON.stringify(v))]));
  const m = d.modulation || {};
  $("mod_head").innerHTML =
    `<span style="font:600 1.3rem var(--mono)">${UI.esc(m.prediction || "-")}</span> ` +
    UI.meter(m.confidence || 0) +
    (m.classifier_agreement === false ? " " + UI.badge("ENGINES DISAGREE", "warn") : "");
  const rows = (m.alternatives || []).map((a) => ["fusion candidate", a[0], a[1]]);
  for (const [eng, pred] of Object.entries(m.engine_predictions || {}))
    rows.push(["engine " + eng, Array.isArray(pred) ? pred[0] : UI.esc(JSON.stringify(pred)),
               Array.isArray(pred) ? pred[1] : ""]);
  for (const c of m.constraints_applied || [])
    rows.push(["constraint", UI.esc(String(c)), ""]);
  UI.kvTable("mod_tbl", rows, ["source", "value", "score"]);
}

function renderBitlayerTab(d) {
  const scr = d.scrambler || {}, il = d.interleaver || {}, fec = d.fec || {};
  UI.kvTable("bl_tbl", [
    ["scrambler", UI.esc(scr.kind || "-"),
     `${UI.esc(scr.name || "")} ${scr.polynomial_hex ? "poly " + scr.polynomial_hex : ""} ` +
     `${scr.seed != null ? "phase " + scr.seed : ""}`],
    ["interleaver", UI.esc(il.kind || "-"),
     `<span class="mono">${UI.esc(JSON.stringify(il.parameters || {}))}</span>`],
    ["FEC", UI.esc(fec.family || "-"),
     `<span class="mono">${UI.esc(JSON.stringify(fec.parameters || {}))}</span> ` +
     (fec.syndrome_zero_rate != null ?
      UI.badge(`szr ${fec.syndrome_zero_rate}`,
               fec.syndrome_zero_rate > 0.9 ? "good" : "warn") : "")],
  ], ["stage", "verdict", "parameters / evidence"]);
  UI.kvTable("hyps", (d.hypotheses || []).map((h) => [
    h.stage, UI.badge(h.status, h.status === "validated" ? "good" :
                      h.status === "rejected" ? "dim" : "info"),
    (h.score ?? 0).toFixed ? h.score.toFixed(3) : h.score,
    `<span class="mono">${UI.esc(JSON.stringify(h.assumptions)).slice(0, 150)}</span>`]),
    ["stage", "status", "score", "assumptions"]);
}

function renderFramesTab(d) {
  const fr = d.frames || {};
  const rows = [
    ["frame length", fr.frame_length_bits != null ? fr.frame_length_bits + " bits" : "-"],
    ["sync word", fr.sync_word_hex ?
      `<span class="mono">${UI.esc(fr.sync_word_hex)}</span>` : "-"],
    ["sync offset", fr.sync_offset],
    ["CRC", fr.crc ? `${fr.crc.name}, poly ${fr.crc.poly}, ` +
      `passes ${fr.crc.passes}/${fr.crc.total}` : "none found"],
  ];
  for (const f of fr.field_map || [])
    rows.push([`bits ${f.start_bit}-${f.start_bit + f.length_bits - 1}`,
               `${UI.esc(f.role)} (entropy ${f.mean_entropy})`]);
  UI.kvTable("frames_tbl", rows);
  const pl = d.payload || {};
  $("pl_meta").innerHTML =
    `${pl.n_bytes ?? 0} bytes &middot; entropy ${pl.entropy_bits_per_bit ?? "-"} b/b ` +
    (pl.likely_encrypted ? UI.badge("HIGH ENTROPY", "warn") : "");
  if (pl.hex)
    UI.hexdump("hexdump", pl.hex.match(/.{2}/g).map((h) => parseInt(h, 16)));
}

function renderLogTab(d) {
  UI.kvTable("warns", (d.warnings || []).map((w) => [UI.esc(w)]));
  UI.kvTable("timings", Object.entries(d.stage_timings || {})
    .map(([k, v]) => [k, v + " s"]));
  const blob = new Blob([JSON.stringify(d, null, 2)], { type: "application/json" });
  $("dl").href = URL.createObjectURL ? URL.createObjectURL(blob) : "#";
  const raw = JSON.stringify(d, null, 1);
  $("rawjson").textContent = raw.length > 200000 ?
    raw.slice(0, 200000) + "\n... truncated for display" : raw;
}

/* charts are drawn lazily when their tab first shows (canvas needs
 * layout), and redrawn on resize */
const drawn = new Set();
function drawCharts(pane) {
  if (!REPORT) return;
  const d = REPORT, plots = d.plots || {};
  if (pane === "overview") return;
  if (drawn.has(pane)) return;
  if (pane === "spectrum") {
    if (plots.psd) UI.lineChart("psd", plots.psd.freq_norm, plots.psd.psd_db,
      { xlabel: "normalised frequency" });
    if (plots.waterfall && plots.waterfall.db) UI.heatmap("wf", plots.waterfall.db);
  } else if (pane === "demod") {
    if (plots.constellation)
      UI.scatterChart("const", plots.constellation.i, plots.constellation.q);
    if (plots.eye) UI.eyeChart("eye", plots.eye.traces);
  } else if (pane === "bitlayer") {
    const rp = plots.rank_profile;
    if (rp && rp.L) UI.lineChart("rank", rp.L, rp.significance,
      { color: UI.COL.series2, xlabel: "trial row length L (bits)" });
  } else if (pane === "frames") {
    const ent = (d.frames || {}).column_entropy || plots.entropy_map;
    if (ent) UI.barChart("entropy", ent, { max: 1, xlabel: "bit position in frame",
      index: (i) => "bit " + i });
  } else return;
  drawn.add(pane);
}
window.addEventListener("resize", () => { drawn.clear(); });
