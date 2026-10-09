/* UI glue for the in-browser analysis page. */
"use strict";

const $ = UI.$;
let currentBuffer = null, currentName = null, lastResult = null;
const tabs = UI.initTabs("tabs", (name) => requestAnimationFrame(() => drawCharts(name)));
const drawn = new Set();

(function setup() {
  const drop = $("drop");
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".wav,.iq,.bin,.raw,.sigmf-data,.cf32,.cs16,.cu8";
  input.style.display = "none";
  drop.appendChild(input);
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files.length && load(input.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("hover"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("hover");
    if (e.dataTransfer.files.length) load(e.dataTransfer.files[0]);
  });
  for (const id of ["ov_rate", "ov_cf", "ov_dtype"])
    $(id).addEventListener("change", () => currentBuffer && run(0));
})();

function load(f) {
  const r = new FileReader();
  $("status").innerHTML = "reading file<span class='cursor'></span>";
  r.onload = () => {
    currentBuffer = r.result;
    currentName = f.name;
    $("drop").innerHTML = `<div class="big">${UI.esc(f.name)}</div>` +
      `${(f.size / 1e6).toFixed(1)} MB loaded &mdash; drop another file to replace`;
    run(0);
  };
  r.readAsArrayBuffer(f.slice(0, 256 * 1024 * 1024));
}

function overrides(signalIndex) {
  const ov = { signalIndex: signalIndex || 0 };
  if ($("ov_rate").value) ov.sampleRate = +$("ov_rate").value;
  if ($("ov_cf").value) ov.centerFrequency = +$("ov_cf").value;
  if ($("ov_dtype").value !== "auto") ov.datatype = $("ov_dtype").value;
  return ov;
}

function run(signalIndex) {
  $("status").innerHTML = "analysing<span class='cursor'></span>";
  setTimeout(() => {
    try {
      const t0 = performance.now();
      const res = analyzeRecording(currentBuffer, currentName,
        overrides(signalIndex),
        (msg) => { $("status").textContent = msg; });
      lastResult = res;
      $("status").innerHTML =
        `analysis finished in ${((performance.now() - t0) / 1000).toFixed(1)} s ` +
        `&mdash; S0-S5 in-browser; the bit layer (S6-S12) runs in the desktop engine`;
      render(res);
    } catch (e) {
      $("status").textContent = "analysis failed: " + e.message;
      console.error(e);
    }
  }, 30);
}

function render(r) {
  $("results").style.display = "block";
  drawn.clear();
  const rec = r.recording;

  UI.renderPipeline("flow", r.pipeline_trace || [], {
    detail: (sid) => stageDetail(sid, r), stepMs: 300 });

  $("in_tiles").innerHTML =
    UI.tile("format", rec.format,
            (rec.datatype || "") +
            (rec.datatypeSource ? ` (${rec.datatypeSource})` : ""), true) +
    UI.tile("sample rate", rec.sampleRate ?
            UI.html(`${(rec.sampleRate / 1e6).toFixed(3)} <small>Msps</small>`) :
            "unknown", rec.sampleRateSource || "") +
    UI.tile("samples", rec.nSamples.toLocaleString(),
            rec.duration ? rec.duration.toFixed(4) + " s" : "") +
    UI.tile("signals found", String(r.signals.length),
            r.signals.length ? `best SNR ${r.signals[0].snrDb} dB` : "");

  UI.kvTable("t_rec", [
    ["file", currentName],
    ["format", rec.format],
    ["datatype", UI.html(UI.esc(rec.datatype) +
      (rec.datatypeSource ? ` <span class="mono">(${UI.esc(rec.datatypeSource)})</span>` : ""))],
    ["sample rate", rec.sampleRate ?
      `${rec.sampleRate.toLocaleString()} Hz (${rec.sampleRateSource})` :
      "unknown - set it in the overrides for absolute units"],
    ["centre frequency", rec.centerFrequency ?
      rec.centerFrequency.toLocaleString() + " Hz" : "-"],
    ["duration", rec.duration ? rec.duration.toFixed(4) + " s" : "-"],
  ]);
  UI.kvTable("t_sniff", (rec.sniff || []).slice(0, 5).map((s) =>
    [s.dtype, UI.html(UI.meter(s.confidence)), s.explanation]),
    rec.sniff ? ["candidate", "confidence", "evidence"] :
    [["header-declared format", "", ""]]);
  const c = r.conditioning;
  UI.kvTable("t_cond", [
    ["DC offset removed", UI.html(`<span class="mono">I ${c.dcOffset[0]}, Q ${c.dcOffset[1]}</span>`)],
    ["clipping", UI.html(`${(c.clippingFraction * 100).toFixed(3)}%` +
      (c.clippingFraction > 1e-3 ? " " + UI.badge("CLIPPED", "warn") : ""))],
    ["normalisation", "unit RMS"],
  ]);

  const t = UI.kvTable("t_signals", r.signals.map((s) => [
    s.id, (+s.center).toFixed(4), (+s.bandwidth).toFixed(4),
    rec.sampleRate ? Math.round(s.center * rec.sampleRate).toLocaleString() : "-",
    rec.sampleRate ? Math.round(s.bandwidth * rec.sampleRate).toLocaleString() : "-",
    `${s.snrDb} dB`,
  ]), ["id", "centre (norm)", "bw (norm)", "centre Hz", "bw Hz", "SNR"]);
  Array.from(t.rows).forEach((row, i) => {
    if (!i) return;
    row.classList.add("clickable");
    if (i - 1 === r.selectedSignal) row.classList.add("selected");
    row.addEventListener("click", () => run(i - 1));
  });

  const p = r.parameters || {};
  $("p_tiles").innerHTML =
    UI.tile("symbol rate", p.symbol_rate_norm != null ?
            (p.symbol_rate_hz ?
             UI.html(`${(p.symbol_rate_hz / 1e3).toFixed(2)} <small>kBd</small>`) :
             UI.html(`${UI.esc(p.symbol_rate_norm)} <small>norm</small>`)) : "not found",
            p.symbol_rate_confidence != null ?
            `confidence ${p.symbol_rate_confidence}` : "", true) +
    UI.tile("SNR (M2M4)", UI.html(`${UI.esc(p.snr_db ?? "-")} <small>dB</small>`), "") +
    UI.tile("occupied bw", `${p.obw99_norm ?? "-"}`, "99% power, normalised") +
    UI.tile("envelope cv", `${p.envelope_cv ?? "-"}`,
            "FSK <= 0.13 | PSK/QAM >= 0.27");
  UI.kvTable("t_params", [
    ["carrier offset (norm)", p.carrier_offset_norm],
    ["samples per symbol", p.samples_per_symbol],
    ["symbol rate (norm)", p.symbol_rate_norm],
    ["symbol rate (Hz)", p.symbol_rate_hz ? p.symbol_rate_hz.toLocaleString() : "-"],
    ...(p.fsk ? [["FSK tones",
      `${p.fsk.toneCount} tones, deviation ${p.fsk.deviation} norm, ` +
      `mass ${p.fsk.massFraction}`]] : []),
  ]);

  const m = r.modulation || {};
  $("mod_verdict").innerHTML =
    `<span style="font:700 1.6rem var(--mono)">${UI.esc(m.prediction || "-")}</span> ` +
    UI.meter(m.confidence || 0) +
    (m.constraints_applied || []).map((cst) =>
      `<div style="color:var(--ink2);font-size:0.82rem;margin-top:5px">` +
      `${UI.esc(cst)}</div>`).join("");
  const modRows = (m.alternatives || []).map((a) => ["candidate", a[0], UI.html(UI.meter(a[1]))]);
  for (const [eng, pred] of Object.entries(m.engine_predictions || {}))
    modRows.push(["engine " + eng,
      Array.isArray(pred) ? pred[0] : UI.html(`<span class="mono">${UI.esc(JSON.stringify(pred))}</span>`),
      Array.isArray(pred) ? pred[1] : ""]);
  UI.kvTable("t_mod", modRows, ["source", "prediction", "score"]);
  UI.kvTable("t_warn", (r.warnings || []).map((w) => [w]));

  const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json" });
  $("dl_json").href = URL.createObjectURL ? URL.createObjectURL(blob) : "#";
  tabs.show("pipeline");
}

function stageDetail(sid, r) {
  const mk = (rows, headers) => {
    if (!rows || !rows.length) return null;
    const t = UI.el("table", "kv");
    UI.kvTable(t, rows.slice(0, 12), headers);
    return t;
  };
  if (sid === "S0") return mk((r.recording.sniff || []).map((s) =>
    [s.dtype, s.confidence, s.explanation]));
  if (sid === "S1") return mk([["clipping", r.conditioning.clippingFraction],
    ["dc", UI.html(`<span class="mono">${r.conditioning.dcOffset.join(", ")}</span>`)]]);
  if (sid === "S2") return mk(r.signals.map((s) =>
    [`signal ${s.id}`, `centre ${s.center.toFixed(4)}, bw ${s.bandwidth.toFixed(4)}, SNR ${s.snrDb} dB`]));
  if (sid === "S4") return mk(Object.entries(r.parameters || {})
    .filter(([k, v]) => v !== null && typeof v !== "object")
    .map(([k, v]) => [k, v]));
  if (sid === "S5") {
    const m = r.modulation || {};
    return mk([["prediction", `${m.prediction} (${m.confidence})`],
      ...(m.alternatives || []).map((a) => ["candidate", `${a[0]}: ${a[1]}`])]);
  }
  return null;
}

function drawCharts(pane) {
  if (!lastResult || drawn.has(pane)) return;
  const plots = lastResult.plots || {};
  if (pane === "spectrum") {
    if (plots.psd) UI.lineChart("psd", plots.psd.freq_norm, plots.psd.psd_db,
      { xlabel: "normalised frequency" });
    if (plots.waterfall) UI.heatmap("wf", plots.waterfall.db);
  } else if (pane === "modulation") {
    if (plots.constellation)
      UI.scatterChart("const", plots.constellation.i, plots.constellation.q);
  } else return;
  drawn.add(pane);
}
window.addEventListener("resize", () => drawn.clear());
