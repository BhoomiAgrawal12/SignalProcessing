/* UI glue for analyze.html: file handling, rendering, signal re-selection. */
"use strict";

const $ = (id) => document.getElementById(id);
let currentBuffer = null, currentName = null, lastResult = null;

/* ---------- canvas helpers (shared style with viewer.js) ---------- */
function ctx2d(id) {
  const c = $(id);
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || c.parentElement.clientWidth || 600;
  c.width = w * dpr;
  c.height = c.getAttribute("height") * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  return [ctx, w, +c.getAttribute("height")];
}

function lineChart(id, xs, ys, color) {
  const [ctx, W, H] = ctx2d(id);
  ctx.clearRect(0, 0, W, H);
  if (!xs || !xs.length) return;
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const px = (x) => 36 + (x - xmin) / (xmax - xmin || 1) * (W - 46);
  const py = (y) => H - 22 - (y - ymin) / (ymax - ymin || 1) * (H - 34);
  ctx.strokeStyle = "#30363d";
  ctx.strokeRect(36, 8, W - 46, H - 30);
  ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.beginPath();
  for (let i = 0; i < xs.length; i++)
    i ? ctx.lineTo(px(xs[i]), py(ys[i])) : ctx.moveTo(px(xs[i]), py(ys[i]));
  ctx.stroke();
  ctx.fillStyle = "#9aa7b4"; ctx.font = "11px sans-serif";
  ctx.fillText(xmin.toPrecision(3), 36, H - 8);
  ctx.fillText(xmax.toPrecision(3), W - 60, H - 8);
  ctx.fillText(ymax.toPrecision(3), 2, 16);
  ctx.fillText(ymin.toPrecision(3), 2, H - 24);
}

function heatmap(id, matrix) {
  const [ctx, W, H] = ctx2d(id);
  ctx.clearRect(0, 0, W, H);
  if (!matrix || !matrix.length) return;
  const rows = matrix.length, cols = matrix[0].length;
  let lo = Infinity, hi = -Infinity;
  for (const row of matrix) for (const v of row) {
    if (v < lo) lo = v; if (v > hi) hi = v;
  }
  const img = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = (matrix[r][c] - lo) / (hi - lo || 1);
      const i = (r * cols + c) * 4;
      img.data[i] = Math.min(255, 510 * Math.max(0, t - 0.5));
      img.data[i + 1] = Math.min(255, 380 * t);
      img.data[i + 2] = 90 + 165 * Math.min(1, t * 2) - 120 * Math.max(0, t - 0.6);
      img.data[i + 3] = 255;
    }
  }
  const off = new OffscreenCanvas(cols, rows);
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, cols, rows, 0, 0, W, H);
}

function scatter(id, xs, ys, color) {
  const [ctx, W, H] = ctx2d(id);
  ctx.clearRect(0, 0, W, H);
  if (!xs || !xs.length) return;
  const m = Math.max(...xs.map(Math.abs), ...ys.map(Math.abs)) * 1.2 || 1;
  const px = (x) => W / 2 + x / m * (Math.min(W, H) / 2 - 10);
  const py = (y) => H / 2 - y / m * (Math.min(W, H) / 2 - 10);
  ctx.strokeStyle = "#30363d";
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
  ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
  ctx.fillStyle = color; ctx.globalAlpha = 0.45;
  for (let i = 0; i < xs.length; i++)
    ctx.fillRect(px(xs[i]) - 1.2, py(ys[i]) - 1.2, 2.4, 2.4);
  ctx.globalAlpha = 1;
}

function kvTable(id, rows, headers) {
  const t = $(id);
  t.innerHTML = "";
  if (headers) {
    const tr = document.createElement("tr");
    for (const h of headers) {
      const th = document.createElement("th"); th.textContent = h;
      tr.appendChild(th);
    }
    t.appendChild(tr);
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const v of row) {
      const td = document.createElement("td");
      td.textContent = v === null || v === undefined ? "-" : String(v);
      tr.appendChild(td);
    }
    t.appendChild(tr);
  }
  return t;
}

function badge(text, cls) {
  return `<span class="badge ${cls}">${text}</span>`;
}

/* ---------- flow ---------- */
function setupDrop() {
  const drop = $("drop"), input = $("file");
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files.length) loadFile(input.files[0]);
  });
  drop.addEventListener("dragover", (e) => {
    e.preventDefault(); drop.classList.add("hover");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("hover");
    if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
  });
  for (const id of ["ov_rate", "ov_cf", "ov_dtype"]) {
    $(id).addEventListener("change", () => {
      if (currentBuffer) run(0);
    });
  }
}

function loadFile(f) {
  if (f.size > 256 * 1024 * 1024) {
    alert("File larger than 256 MB: the browser engine analyses the first " +
          "part only; for huge files use the desktop tool (memory-mapped).");
  }
  const r = new FileReader();
  $("status").textContent = "reading file...";
  r.onload = () => {
    currentBuffer = r.result;
    currentName = f.name;
    $("drop").textContent = `${f.name} (${(f.size / 1e6).toFixed(1)} MB) - ` +
      "drop another file to replace";
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
  $("status").textContent = "analysing...";
  setTimeout(() => {
    try {
      const t0 = performance.now();
      const res = analyzeRecording(currentBuffer, currentName,
        overrides(signalIndex),
        (msg) => { $("status").textContent = msg; });
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      lastResult = res;
      $("status").textContent = `analysis finished in ${dt} s ` +
        "(S0-S5 in-browser; bit layer runs in the desktop tool)";
      render(res);
    } catch (e) {
      $("status").textContent = "analysis failed: " + e.message;
      console.error(e);
    }
  }, 30);
}

function render(r) {
  $("results").style.display = "block";
  const rec = r.recording;
  kvTable("t_rec", [
    ["file", currentName],
    ["format", rec.format],
    ["datatype", rec.datatype + (rec.datatypeSource ? ` (${rec.datatypeSource})` : "")],
    ["sample rate", rec.sampleRate
      ? `${rec.sampleRate} Hz (${rec.sampleRateSource})`
      : "unknown - normalised units in use; set it above for Hz"],
    ["centre frequency", rec.centerFrequency || "-"],
    ["samples analysed", rec.nSamples],
    ["duration", rec.duration ? rec.duration.toFixed(4) + " s" : "-"],
  ]);
  kvTable("t_sniff", (rec.sniff || []).slice(0, 4).map((s) =>
    [s.dtype, s.confidence, s.explanation]),
    rec.sniff ? ["candidate", "confidence", "evidence"] : null);

  const c = r.conditioning;
  kvTable("t_cond", [
    ["DC offset removed", `I ${c.dcOffset[0]}, Q ${c.dcOffset[1]}`],
    ["clipping fraction", c.clippingFraction],
    ["normalised to unit RMS", c.normalised],
  ]);

  if (r.plots.psd) lineChart("psd", r.plots.psd.freq_norm, r.plots.psd.psd_db, "#4fc3f7");
  if (r.plots.waterfall) heatmap("wf", r.plots.waterfall.db);

  const t = kvTable("t_signals", r.signals.map((s) => [
    s.id + (r.selectedSignal === s.id ? "  (selected)" : ""),
    s.center.toFixed(4), s.bandwidth.toFixed(4),
    rec.sampleRate ? Math.round(s.center * rec.sampleRate) : "-",
    rec.sampleRate ? Math.round(s.bandwidth * rec.sampleRate) : "-",
    s.snrDb,
  ]), ["id", "centre (norm)", "bw (norm)", "centre Hz", "bw Hz", "SNR dB"]);
  Array.from(t.rows).forEach((row, i) => {
    if (i === 0) return;
    row.style.cursor = "pointer";
    row.addEventListener("click", () => run(i - 1));
  });

  const p = r.parameters || {};
  kvTable("t_params", [
    ["SNR (M2M4)", p.snr_db != null ? p.snr_db + " dB" : "-"],
    ["occupied bandwidth (99%)", p.obw99_norm],
    ["carrier offset (norm)", p.carrier_offset_norm],
    ["symbol rate (norm)", p.symbol_rate_norm != null
      ? `${p.symbol_rate_norm} (confidence ${p.symbol_rate_confidence})` : "not found"],
    ["symbol rate (Hz)", p.symbol_rate_hz || "-"],
    ["samples per symbol", p.samples_per_symbol],
    ["envelope cv", p.envelope_cv + "  (FSK <= 0.13, shaped PSK/QAM >= 0.27)"],
    ...(p.fsk ? [["FSK tones", `${p.fsk.toneCount} tones, deviation ` +
                  `${p.fsk.deviation} (norm), mass ${p.fsk.massFraction}`]] : []),
  ]);

  const m = r.modulation || {};
  let v = `<p style="font-size:1.4rem;font-weight:700">${m.prediction || "-"}` +
    badge(`confidence ${m.confidence ?? "-"}`,
          (m.confidence || 0) > 0.7 ? "good" : "warn") + "</p>";
  for (const cst of m.constraints_applied || [])
    v += `<p style="color:var(--dim);font-size:0.88rem">${cst}</p>`;
  $("mod_verdict").innerHTML = v;
  const modRows = (m.alternatives || []).map((a) => ["candidate", a[0], a[1]]);
  for (const [eng, pred] of Object.entries(m.engine_predictions || {})) {
    if (Array.isArray(pred)) modRows.push(["engine " + eng, pred[0], pred[1]]);
    else modRows.push(["engine " + eng, JSON.stringify(pred), ""]);
  }
  kvTable("t_mod", modRows, ["source", "prediction", "score"]);

  if (r.plots.constellation)
    scatter("const", r.plots.constellation.i, r.plots.constellation.q, "#4fc3f7");
  else {
    const [ctx, W, H] = ctx2d("const");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#9aa7b4";
    ctx.fillText("no symbol-spaced preview (FSK or no symbol rate)", 20, H / 2);
  }

  kvTable("t_warn", (r.warnings || []).map((w) => [w]));

  const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json" });
  $("dl_json").href = URL.createObjectURL(blob);
}

setupDrop();
