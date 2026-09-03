/* rf-analyzer report viewer: renders analysis.json entirely client-side. */
"use strict";

const $ = (id) => document.getElementById(id);

function setupDrop() {
  const drop = $("drop"), input = $("file");
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files.length) readFile(input.files[0]);
  });
  drop.addEventListener("dragover", (e) => {
    e.preventDefault(); drop.classList.add("hover");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("hover");
    if (e.dataTransfer.files.length) readFile(e.dataTransfer.files[0]);
  });
}

function readFile(f) {
  const name = (f.name || "").toLowerCase();
  if (!name.endsWith(".json")) {
    if (confirm("This page views exported analysis.json reports. " +
        "To analyse a raw .wav/.iq recording in the browser, use the " +
        "Analyse page instead. Go there now?")) {
      location.href = "analyze.html";
    }
    return;
  }
  const r = new FileReader();
  r.onload = () => {
    try { render(JSON.parse(r.result)); }
    catch (err) { alert("Not a valid analysis JSON: " + err); }
  };
  r.readAsText(f);
}

/* ---------- canvas helpers ---------- */
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
  for (let i = 0; i < xs.length; i++) {
    i ? ctx.lineTo(px(xs[i]), py(ys[i])) : ctx.moveTo(px(xs[i]), py(ys[i]));
  }
  ctx.stroke();
  ctx.fillStyle = "#9aa7b4"; ctx.font = "11px sans-serif";
  ctx.fillText(xmin.toPrecision(3), 36, H - 8);
  ctx.fillText(xmax.toPrecision(3), W - 60, H - 8);
  ctx.fillText(ymax.toPrecision(3), 2, 16);
  ctx.fillText(ymin.toPrecision(3), 2, H - 24);
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
  ctx.fillStyle = color;
  for (let i = 0; i < xs.length; i++) {
    ctx.globalAlpha = 0.45;
    ctx.fillRect(px(xs[i]) - 1.2, py(ys[i]) - 1.2, 2.4, 2.4);
  }
  ctx.globalAlpha = 1;
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
      /* dark blue -> cyan -> yellow */
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

function eyeChart(id, traces) {
  const [ctx, W, H] = ctx2d(id);
  ctx.clearRect(0, 0, W, H);
  if (!traces || !traces.length) return;
  let m = 0;
  for (const t of traces) for (const v of t) m = Math.max(m, Math.abs(v));
  m = m * 1.15 || 1;
  ctx.strokeStyle = "rgba(79,195,247,0.25)";
  for (const t of traces) {
    ctx.beginPath();
    for (let i = 0; i < t.length; i++) {
      const x = i / (t.length - 1) * (W - 10) + 5;
      const y = H / 2 - t[i] / m * (H / 2 - 8);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
}

function bars(id, values, color) {
  const [ctx, W, H] = ctx2d(id);
  ctx.clearRect(0, 0, W, H);
  if (!values || !values.length) return;
  const bw = (W - 12) / values.length;
  ctx.fillStyle = color;
  for (let i = 0; i < values.length; i++) {
    const h = values[i] * (H - 18);
    ctx.fillRect(6 + i * bw, H - 10 - h, Math.max(1, bw - 0.5), h);
  }
  ctx.strokeStyle = "#30363d"; ctx.strokeRect(6, 6, W - 12, H - 16);
}

/* ---------- tables ---------- */
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
}

function badge(text, cls) {
  return `<span class="badge ${cls}">${text}</span>`;
}

/* ---------- main render ---------- */
function render(d) {
  $("drop").style.display = "none";
  $("content").style.display = "block";

  const mod = d.modulation || {};
  const fec = d.fec || {};
  const fr = d.frames || {};
  const pl = d.payload || {};
  const crc = fr.crc || null;
  let v = `<p>Modulation <strong>${mod.prediction || "unknown"}</strong>` +
    badge(`confidence ${mod.confidence ?? "-"}`,
          (mod.confidence || 0) > 0.7 ? "good" : "warn");
  if (mod.classifier_agreement === false)
    v += badge("engines disagree", "warn");
  v += `</p><p>FEC <strong>${fec.family || "-"}</strong>`;
  if (fec.syndrome_zero_rate != null)
    v += badge(`syndrome-zero rate ${(fec.syndrome_zero_rate * 100).toFixed(1)}%`,
               fec.syndrome_zero_rate > 0.9 ? "good" : "warn");
  v += ` | interleaver <strong>${(d.interleaver || {}).kind || "-"}</strong>`;
  v += ` | scrambler <strong>${(d.scrambler || {}).name || (d.scrambler || {}).kind || "-"}</strong></p>`;
  if (crc) {
    v += `<p>CRC ${crc.name} ` +
      badge(`pass ${(crc.pass_fraction * 100).toFixed(0)}%`,
            crc.pass_fraction > 0.9 ? "good" : "warn") + "</p>";
  } else {
    v += `<p>CRC ${badge("none found", "warn")}</p>`;
  }
  if (pl.likely_encrypted)
    v += `<p>${badge("payload entropy near 1 bit/bit: likely encrypted - no recovery attempted", "bad")}</p>`;
  $("verdict").innerHTML = v;

  const plots = d.plots || {};
  if (plots.psd) lineChart("psd", plots.psd.freq_norm, plots.psd.psd_db, "#4fc3f7");
  if (plots.waterfall && plots.waterfall.db) heatmap("wf", plots.waterfall.db);
  if (plots.constellation) scatter("const", plots.constellation.i, plots.constellation.q, "#4fc3f7");
  if (plots.eye) eyeChart("eye", plots.eye.traces);
  if (plots.rank_profile && plots.rank_profile.L)
    lineChart("rank", plots.rank_profile.L, plots.rank_profile.significance, "#ffb74d");
  const ent = fr.column_entropy || plots.entropy_map;
  if (ent) bars("entropy", ent, "#4fc3f7");

  const rec = d.recording || {};
  const p = d.parameters || {};
  const conf = p.confidences || {};
  const c = (n) => (conf[n] || {});
  kvTable("params", [
    ["file", rec.file_path, "", ""],
    ["format", rec.format, "", rec.datatype],
    ["sample rate", rec.sample_rate ?? "unknown (headerless raw IQ)", "", rec.sample_rate_source],
    ["duration", rec.duration_s ? rec.duration_s.toFixed(4) + " s" : "-", "", ""],
    ["SNR", p.snr_db != null ? p.snr_db + " dB" : "-", c("snr").value, c("snr").method],
    ["occupied bandwidth (99%)", p.obw99_norm, c("obw").value, c("obw").method],
    ["carrier offset (norm)", p.carrier_offset_norm, c("cfo").value, c("cfo").method],
    ["symbol rate (norm)", p.symbol_rate_norm, c("symbol_rate").value, c("symbol_rate").method],
    ["symbol rate (Hz)", p.symbol_rate_hz, "", ""],
    ["samples/symbol", p.samples_per_symbol, "", ""],
    ["excess bandwidth", p.excess_bandwidth, "", ""],
  ], ["parameter", "value", "confidence", "method"]);

  const modRows = (mod.alternatives || []).map((a) => ["fusion", a[0], a[1]]);
  for (const [eng, pred] of Object.entries(mod.engine_predictions || {})) {
    if (Array.isArray(pred)) modRows.push(["engine " + eng, pred[0], pred[1]]);
    else modRows.push(["engine " + eng, JSON.stringify(pred), ""]);
  }
  kvTable("mod", modRows, ["source", "prediction", "score"]);

  kvTable("bitlayer", [
    ["scrambler", (d.scrambler || {}).kind, JSON.stringify({
      name: (d.scrambler || {}).name, poly: (d.scrambler || {}).polynomial_hex,
      phase: (d.scrambler || {}).seed })],
    ["interleaver", (d.interleaver || {}).kind,
     JSON.stringify((d.interleaver || {}).parameters)],
    ["FEC", fec.family, JSON.stringify(fec.parameters)],
    ["FEC evidence", "syndrome-zero rate", fec.syndrome_zero_rate],
  ], ["stage", "verdict", "details"]);

  const frRows = [
    ["frame length", fr.frame_length_bits != null ? fr.frame_length_bits + " bits" : "-"],
    ["sync word", fr.sync_word_hex],
    ["CRC", crc ? `${crc.name} (poly ${crc.poly}, pass ${crc.passes}/${crc.total})` : "none"],
  ];
  for (const f of (fr.field_map || []))
    frRows.push([`field @ bit ${f.start_bit}`,
                 `${f.role}, ${f.length_bits} bits, entropy ${f.mean_entropy}`]);
  kvTable("frames", frRows);

  $("payload_meta").innerHTML =
    `<p>${pl.n_bytes ?? 0} bytes | entropy ${pl.entropy_bits_per_bit ?? "-"} bits/bit | ` +
    `printable ${pl.printable_fraction != null ? (pl.printable_fraction * 100).toFixed(1) + "%" : "-"}</p>`;
  if (pl.hex) {
    const bytes = pl.hex.match(/.{2}/g).slice(0, 2048).map((h) => parseInt(h, 16));
    const lines = [];
    for (let i = 0; i < bytes.length; i += 16) {
      const chunk = bytes.slice(i, i + 16);
      const hx = chunk.map((b) => b.toString(16).padStart(2, "0")).join(" ");
      const asc = chunk.map((b) => (b >= 32 && b < 127 ? String.fromCharCode(b) : ".")).join("");
      lines.push(i.toString(16).padStart(8, "0") + "  " + hx.padEnd(48) + "  " + asc);
    }
    $("hexdump").textContent = lines.join("\n");
  } else {
    $("hexdump").textContent = "no payload recovered";
  }

  kvTable("hyps", (d.hypotheses || []).map((h) => [
    h.stage, h.status, h.score, JSON.stringify(h.assumptions).slice(0, 140),
    JSON.stringify(h.evidence).slice(0, 120),
  ]), ["stage", "status", "score", "assumptions", "evidence"]);

  const wRows = (d.warnings || []).map((w) => ["warning", w]);
  for (const [k, secs] of Object.entries(d.stage_timings || {}))
    wRows.push(["timing " + k, secs + " s"]);
  kvTable("warnings", wRows);
}

setupDrop();
if (new URLSearchParams(location.search).get("demo")) {
  fetch("demo/analysis.json").then((r) => r.json()).then(render)
    .catch((e) => alert("demo report unavailable: " + e));
}
