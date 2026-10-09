/* Shared UI library for the Dhwani web console: tabs, canvas charts
 * with hover readouts, tables, badges, the animated pipeline flow, and
 * the payload-intelligence renderer. Plain JS, no dependencies. */
"use strict";

const UI = (() => {

const COL = {
  series1: "#1e9de0", series2: "#d17c07", glow: "#4fc3f7",
  good: "#3fb950", warn: "#d29922", bad: "#f85149",
  ink2: "#8b98ab", muted: "#5f6c80", grid: "#232b3a", surface: "#0a0e13",
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
/* analysis.json is untrusted: tile() and kvTable() escape every value
 * unless the caller explicitly marks a fragment it built as html(). */
const html = (s) => ({ __html: String(s) });
const isHtml = (v) => v !== null && typeof v === "object" && "__html" in v;
const frag = (v) => isHtml(v) ? v.__html : esc(v);

/* ---------------- tabs ---------------- */
function initTabs(navId, onShow) {
  const nav = $(navId);
  const buttons = Array.from(nav.querySelectorAll("button"));
  function show(name) {
    for (const b of buttons) b.classList.toggle("active", b.dataset.tab === name);
    for (const p of document.querySelectorAll(".tabpane"))
      p.classList.toggle("active", p.dataset.pane === name);
    if (onShow) onShow(name);
  }
  for (const b of buttons)
    b.addEventListener("click", () => !b.disabled && show(b.dataset.tab));
  return { show };
}

/* ---------------- badges / meters ---------------- */
function badge(text, kind) {
  return `<span class="badge ${kind}">${esc(text)}</span>`;
}
function strengthBadge(strength) {
  const kind = { VALIDATED: "good", LIKELY: "info", POSSIBLE: "warn",
                 WEAK: "dim" }[strength] || "dim";
  return badge(strength, kind);
}
function meter(v) {
  const pct = Math.round(Math.max(0, Math.min(1, Number(v) || 0)) * 100);
  return `<span class="meter"><span class="bar"><i style="width:${pct}%"></i></span>` +
         `<span class="num">${esc(Number(v ?? 0).toFixed(2))}</span></span>`;
}
function tile(k, v, sub, accent) {
  return `<div class="tile${accent ? " accent" : ""}"><div class="k">${esc(k)}</div>` +
         `<div class="v">${frag(v)}</div>${sub ? `<div class="s">${frag(sub)}</div>` : ""}</div>`;
}

/* ---------------- tables ---------------- */
function kvTable(target, rows, headers) {
  const t = typeof target === "string" ? $(target) : target;
  t.innerHTML = "";
  if (headers) {
    const tr = el("tr");
    for (const h of headers) tr.appendChild(el("th", null, esc(h)));
    t.appendChild(tr);
  }
  for (const row of rows) {
    const tr = el("tr");
    for (const v of row) {
      const td = el("td");
      if (isHtml(v)) td.innerHTML = v.__html;
      else td.textContent = v === null || v === undefined ? "\u2013" : String(v);
      tr.appendChild(td);
    }
    t.appendChild(tr);
  }
  return t;
}

/* ---------------- canvas helpers ---------------- */
function prep(canvas) {
  const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth || 640;
  const h = +canvas.getAttribute("height") || 240;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext && canvas.getContext("2d");
  if (!ctx) return [null, w, h];      /* headless / unsupported canvas */
  ctx.scale(dpr, dpr);
  return [ctx, w, h];
}

function lineChart(canvas, xs, ys, opts = {}) {
  canvas = typeof canvas === "string" ? $(canvas) : canvas;
  const [ctx, W, H] = prep(canvas);
  if (!ctx) return;
  ctx.clearRect(0, 0, W, H);
  if (!xs || xs.length < 2) return;
  const padL = 44, padR = 10, padT = 10, padB = 24;
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const px = (x) => padL + (x - xmin) / (xmax - xmin || 1) * (W - padL - padR);
  const py = (y) => H - padB - (y - ymin) / (ymax - ymin || 1) * (H - padT - padB);
  // recessive grid
  ctx.strokeStyle = COL.grid; ctx.lineWidth = 1;
  ctx.font = "10px ui-monospace, monospace"; ctx.fillStyle = COL.muted;
  for (let g = 0; g <= 4; g++) {
    const gy = padT + g * (H - padT - padB) / 4;
    ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(W - padR, gy); ctx.stroke();
    const val = ymax - g * (ymax - ymin) / 4;
    ctx.fillText(val.toPrecision(3), 4, gy + 3);
  }
  ctx.fillText(xmin.toPrecision(3), padL, H - 8);
  const xmaxLabel = xmax.toPrecision(3);
  ctx.fillText(xmaxLabel, W - padR - ctx.measureText(xmaxLabel).width, H - 8);
  if (opts.xlabel) {
    ctx.fillText(opts.xlabel, (W - ctx.measureText(opts.xlabel).width) / 2, H - 8);
  }
  ctx.strokeStyle = opts.color || COL.series1;
  ctx.lineWidth = 1.6; ctx.beginPath();
  for (let i = 0; i < xs.length; i++)
    i ? ctx.lineTo(px(xs[i]), py(ys[i])) : ctx.moveTo(px(xs[i]), py(ys[i]));
  ctx.stroke();
  attachHover(canvas, (mx) => {
    // nearest x readout
    let best = 0, bd = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(px(xs[i]) - mx);
      if (d < bd) { bd = d; best = i; }
    }
    return { x: px(xs[best]), y: py(ys[best]),
             text: `${xs[best].toPrecision(4)} : ${ys[best].toPrecision(4)}` };
  });
}

function barChart(canvas, values, opts = {}) {
  canvas = typeof canvas === "string" ? $(canvas) : canvas;
  const [ctx, W, H] = prep(canvas);
  if (!ctx) return;
  ctx.clearRect(0, 0, W, H);
  if (!values || !values.length) return;
  const padL = 30, padB = 18, padT = 8;
  const vmax = opts.max ?? Math.max(...values, 1e-9);
  const bw = (W - padL - 8) / values.length;
  ctx.fillStyle = opts.color || COL.series1;
  for (let i = 0; i < values.length; i++) {
    const h = Math.max(1, values[i] / vmax * (H - padT - padB));
    const x = padL + i * bw, y = H - padB - h;
    const w = Math.max(1, bw - (bw > 3 ? 1 : 0.2));
    if (bw >= 4) {                       // rounded data-end
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, [2, 2, 0, 0]);
      ctx.fill();
    } else ctx.fillRect(x, y, w, h);
  }
  ctx.strokeStyle = COL.grid;
  ctx.strokeRect(padL, padT, W - padL - 8, H - padT - padB);
  ctx.fillStyle = COL.muted; ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(vmax.toPrecision(2), 2, padT + 8);
  ctx.fillText("0", 2, H - padB);
  if (opts.xlabel)
    ctx.fillText(opts.xlabel, (W - ctx.measureText(opts.xlabel).width) / 2, H - 4);
  attachHover(canvas, (mx) => {
    const i = Math.max(0, Math.min(values.length - 1,
      Math.floor((mx - padL) / bw)));
    return { x: padL + i * bw + bw / 2,
             y: H - padB - values[i] / vmax * (H - padT - padB),
             text: `${opts.index ? opts.index(i) : i} : ${values[i].toPrecision ? values[i].toPrecision(3) : values[i]}` };
  });
}

function scatterChart(canvas, xs, ys, opts = {}) {
  canvas = typeof canvas === "string" ? $(canvas) : canvas;
  const [ctx, W, H] = prep(canvas);
  if (!ctx) return;
  ctx.clearRect(0, 0, W, H);
  if (!xs || !xs.length) return;
  const m = (opts.range || Math.max(...xs.map(Math.abs), ...ys.map(Math.abs))) * 1.2 || 1;
  const px = (x) => W / 2 + x / m * (Math.min(W, H) / 2 - 8);
  const py = (y) => H / 2 - y / m * (Math.min(W, H) / 2 - 8);
  ctx.strokeStyle = COL.grid;
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
  ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
  ctx.fillStyle = opts.color || COL.series1;
  ctx.globalAlpha = 0.5;
  for (let i = 0; i < xs.length; i++)
    ctx.fillRect(px(xs[i]) - 1.3, py(ys[i]) - 1.3, 2.6, 2.6);
  ctx.globalAlpha = 1;
}

function heatmap(canvas, matrix) {
  canvas = typeof canvas === "string" ? $(canvas) : canvas;
  const [ctx, W, H] = prep(canvas);
  if (!ctx) return;
  ctx.clearRect(0, 0, W, H);
  if (!matrix || !matrix.length) return;
  const rows = matrix.length, cols = matrix[0].length;
  let lo = Infinity, hi = -Infinity;
  for (const r of matrix) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const img = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const t = (matrix[r][c] - lo) / (hi - lo || 1);
    const i = (r * cols + c) * 4;
    /* single-hue sequential (deep blue -> cyan -> white-hot) */
    img.data[i]     = Math.round(255 * Math.max(0, t - 0.55) / 0.45 * 0.9 + 10 * t);
    img.data[i + 1] = Math.round(30 + 190 * t);
    img.data[i + 2] = Math.round(60 + 195 * Math.min(1, t * 1.5));
    img.data[i + 3] = 255;
  }
  const off = new OffscreenCanvas(cols, rows);
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, cols, rows, 0, 0, W, H);
}

function eyeChart(canvas, traces) {
  canvas = typeof canvas === "string" ? $(canvas) : canvas;
  const [ctx, W, H] = prep(canvas);
  if (!ctx) return;
  ctx.clearRect(0, 0, W, H);
  if (!traces || !traces.length) return;
  let m = 0;
  for (const t of traces) for (const v of t) m = Math.max(m, Math.abs(v));
  m = m * 1.15 || 1;
  ctx.strokeStyle = "rgba(30,157,224,0.22)";
  ctx.lineWidth = 1;
  for (const t of traces) {
    ctx.beginPath();
    for (let i = 0; i < t.length; i++) {
      const x = i / (t.length - 1) * (W - 12) + 6;
      const y = H / 2 - t[i] / m * (H / 2 - 8);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
}

/* hover tooltip layer */
function attachHover(canvas, locate) {
  if (canvas._hover) canvas.removeEventListener("mousemove", canvas._hover);
  let tip = canvas.parentElement.querySelector(".chart-tip");
  if (!tip) {
    tip = el("div", "chart-tip");
    tip.style.cssText = "position:absolute;pointer-events:none;display:none;" +
      "background:#171d29;border:1px solid #33405a;border-radius:4px;" +
      "padding:3px 8px;font:11px ui-monospace,monospace;color:#dbe4f0;z-index:20";
    canvas.parentElement.style.position = "relative";
    canvas.parentElement.appendChild(tip);
  }
  canvas._hover = (ev) => {
    const r = canvas.getBoundingClientRect();
    const hit = locate(ev.clientX - r.left, ev.clientY - r.top);
    if (!hit) { tip.style.display = "none"; return; }
    tip.textContent = hit.text;
    tip.style.display = "block";
    tip.style.left = Math.min(r.width - 130, hit.x + 10) + "px";
    tip.style.top = Math.max(0, hit.y - 26) + "px";
  };
  canvas.addEventListener("mousemove", canvas._hover);
  canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });
}

/* ---------------- hexdump ---------------- */
function hexdump(target, bytes, maxBytes = 2048) {
  const t = typeof target === "string" ? $(target) : target;
  if (!bytes || !bytes.length) { t.textContent = "no payload recovered"; return; }
  const lines = [];
  const n = Math.min(bytes.length, maxBytes);
  for (let i = 0; i < n; i += 16) {
    const chunk = bytes.slice(i, i + 16);
    const hx = Array.from(chunk, (b) => b.toString(16).padStart(2, "0")).join(" ");
    const asc = Array.from(chunk, (b) => b >= 32 && b < 127 ?
      String.fromCharCode(b) : ".").join("");
    lines.push(`<span class="off">${i.toString(16).padStart(8, "0")}</span>  ` +
               `${hx.padEnd(48)}  <span class="asc">${esc(asc)}</span>`);
  }
  if (bytes.length > maxBytes)
    lines.push(`... ${bytes.length - maxBytes} more bytes`);
  t.innerHTML = lines.join("\n");
}

/* ---------------- pipeline flow (animated) ---------------- */
function renderPipeline(target, trace, opts = {}) {
  const wrap = typeof target === "string" ? $(target) : target;
  wrap.innerHTML = "";
  const controls = el("div", "flow-controls");
  const replay = el("button", "btn ghost", "replay");
  const summaryChips = el("div", null,
    `${badge(trace.filter((t) => t.status === "executed").length + " executed", "good")} ` +
    `${badge(trace.filter((t) => t.status === "skipped").length + " skipped", "dim")} ` +
    (trace.some((t) => t.status === "stopped") ? badge("stopped early", "bad") : ""));
  controls.appendChild(summaryChips);
  controls.appendChild(el("div", "spacer"));
  controls.appendChild(replay);
  wrap.appendChild(controls);
  const list = el("div", "flowwrap");
  wrap.appendChild(list);

  const stages = [];
  trace.forEach((t, i) => {
    const row = el("div", `flow-stage ${esc(t.status)}`);
    if (i < trace.length - 1) row.appendChild(el("div", "wire"));
    row.appendChild(el("div", "node"));
    row.appendChild(el("div", "sid", esc(t.stage)));
    const body = el("div", "body");
    body.appendChild(el("div", "title", esc(t.title)));
    body.appendChild(el("div", "summary", esc(t.summary || "")));
    const expand = el("div", "expand");
    if (opts.detail) {
      const d = opts.detail(t.stage);
      if (d) expand.appendChild(d);
    }
    body.appendChild(expand);
    body.addEventListener("click", () => row.classList.toggle("open"));
    row.appendChild(body);
    const meta = el("div", "meta");
    const statusKind = { executed: "good", ready: "info",
                         skipped: "dim", stopped: "bad" }[t.status] || "dim";
    meta.innerHTML = badge(String(t.status).toUpperCase(), statusKind) +
      (t.elapsed_s != null ? `<span class="time">${esc(t.elapsed_s)}s</span>` : "");
    row.appendChild(meta);
    list.appendChild(row);
    stages.push(row);
  });

  let timer = null;
  function play() {
    if (timer) clearInterval(timer);
    const reduced = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    stages.forEach((s) => { s.classList.remove("revealed", "active"); });
    if (reduced) {
      stages.forEach((s) => s.classList.add("revealed"));
      return;
    }
    let i = 0;
    timer = setInterval(() => {
      if (i > 0) stages[i - 1].classList.remove("active");
      if (i >= stages.length) { clearInterval(timer); timer = null; return; }
      stages[i].classList.add("revealed");
      if (trace[i].status === "executed" || trace[i].status === "ready")
        stages[i].classList.add("active");
      i++;
    }, opts.stepMs || 340);
  }
  replay.addEventListener("click", play);
  play();
}

/* ---------------- payload intelligence renderer ---------------- */
function renderFinding(f) {
  const d = el("div", `finding ${f.strength || "WEAK"}`);
  let head = `<span class="cat">${esc(f.category)}</span>` +
    `<span class="verdict">${esc(f.verdict)}</span>` +
    strengthBadge(f.strength) + meter(f.confidence);
  d.appendChild(el("div", "head", head));
  const items = [];
  for (const e of f.evidence || []) items.push(`<li>${esc(e)}</li>`);
  for (const l of f.limitations || [])
    items.push(`<li class="lim">limitation: ${esc(l)}</li>`);
  if (items.length) d.appendChild(el("ul", null, items.join("")));
  const det = f.details || {};
  if (det.preview || det.decoded_preview) {
    d.appendChild(el("div", "mono",
      `&gt; ${esc(String(det.preview || det.decoded_preview).slice(0, 200))}`));
  }
  return d;
}

function renderPayloadIntel(container, pi, payloadHexStr) {
  const c = typeof container === "string" ? $(container) : container;
  c.innerHTML = "";
  if (!pi || pi.available === undefined) {
    c.appendChild(el("div", "panel",
      "<h3>Payload intelligence</h3><p style='color:var(--ink2)'>Payload " +
      "Intelligence not available in this analysis version. Re-run the " +
      "desktop analyzer to add S11 results.</p>"));
    return;
  }
  if (!pi.available) {
    c.appendChild(el("div", "panel",
      `<h3>Payload intelligence</h3><p style='color:var(--ink2)'>` +
      `${esc(pi.reason || "no payload recovered")}</p>`));
    return;
  }
  const bf = pi.byte_forensics || {};
  const summ = pi.summary || {};
  const tiles = el("div", "tiles");
  tiles.innerHTML =
    tile("classification", summ.classification || "-",
         html(strengthBadge(summ.strength) + " " + meter(summ.confidence || 0)), true) +
    tile("payload size", html(`${esc(bf.n_bytes ?? 0)} <small>bytes</small>`),
         `${bf.n_bits ?? "-"} bits`) +
    tile("entropy", html(`${esc(bf.entropy_bits_per_byte ?? "-")} <small>bits/byte</small>`),
         `normalised ${bf.entropy_normalised ?? "-"}`) +
    tile("printable", html(`${esc((Number(bf.printable_ratio ?? 0) * 100).toFixed(1))}<small>%</small>`),
         `${bf.unique_bytes ?? "-"} unique byte values`);
  const p0 = el("div", "panel");
  p0.appendChild(el("h3", null, "S11 &mdash; payload intelligence summary"));
  const trust = (pi.summary || {}).trust || (pi.provenance || {}).trust;
  if (trust) {
    const kind = trust.startsWith("VALIDATED") ? "good" :
      trust.startsWith("PROBABLE") ? "info" : "warn";
    p0.appendChild(el("div", null,
      `<div style="margin-bottom:12px">${badge(trust, kind)}</div>`));
  }
  p0.appendChild(tiles);
  const prov = pi.provenance || {};
  p0.appendChild(el("div", null,
    `<div class="microlabel" style="margin:12px 0 6px">provenance</div>` +
    badge(prov.crc_validated ? "CRC VALIDATED" : "NO CRC VALIDATION",
          prov.crc_validated ? "good" : "warn") + " " +
    (prov.fec_syndrome_rate != null ?
      badge(`FEC syndrome-zero ${prov.fec_syndrome_rate}`, "info") : "") + " " +
    (prov.demod_status ?
      badge(`DEMOD ${prov.demod_status}`,
            prov.demod_status === "GOOD" ? "good" :
            prov.demod_status === "DEGRADED" ? "warn" : "bad") :
     prov.demod_locked != null ?
      badge(prov.demod_locked ? "DEMOD LOCKED" : "WEAK DEMOD LOCK",
            prov.demod_locked ? "good" : "warn") : "")));
  c.appendChild(p0);

  // entropy windows strip
  if ((bf.entropy_windows || []).length > 1) {
    const p = el("div", "panel");
    p.appendChild(el("h3", null,
      "entropy across the payload <span class='hint'>64-byte windows; " +
      "flat high entropy suggests compressed/encrypted, dips are structure</span>"));
    const cv = el("canvas", "chart");
    cv.setAttribute("height", 110);
    p.appendChild(cv);
    c.appendChild(p);
    requestAnimationFrame(() => barChart(cv,
      bf.entropy_windows.map((w) => w.entropy), { max: 1,
        index: (i) => `offset ${bf.entropy_windows[i].offset}` }));
  }

  // field map
  const fields = (pi.structure || {}).fields || [];
  if (fields.length) {
    const p = el("div", "panel");
    p.appendChild(el("h3", null,
      "cross-frame field map <span class='hint'>byte roles inferred " +
      `across ${esc(((pi.structure || {}).cross_frame || {}).n_frames || "?")} frames</span>`));
    const strip = el("div", "fieldmap");
    const total = fields.reduce((s, f) => s + f.length_bytes, 0) || 1;
    for (const f of fields) {
      const seg = el("div", String(f.role).replace("length-field candidate", "length-field"));
      seg.style.width = (Number(f.length_bytes) / total * 100) + "%";
      seg.title = `bytes ${f.start_byte}-${f.start_byte + f.length_bytes - 1}: ` +
        `${f.role}${f.note ? " (" + f.note + ")" : ""}, entropy ${f.mean_entropy}`;
      if (f.length_bytes / total > 0.08)
        seg.textContent = `${String(f.role).split("/")[0]} ${f.length_bytes}B`;
      strip.appendChild(seg);
    }
    p.appendChild(strip);
    p.appendChild(el("div", "legend",
      `<span><i style="background:${COL.good}"></i>constant/marker</span>` +
      `<span><i style="background:${COL.series1}"></i>counter</span>` +
      `<span><i style="background:${COL.series2}"></i>length field</span>` +
      `<span><i style="background:#3a4658"></i>variable/payload</span>`));
    const rows = fields.map((f) => [f.start_byte, f.length_bytes,
      f.role, f.mean_entropy, f.note || ""]);
    const tbl = el("table", "kv");
    p.appendChild(tbl);
    kvTable(tbl, rows, ["start byte", "length", "inferred role",
                        "entropy", "note"]);
    c.appendChild(p);
  }

  // findings columns
  const cols = el("div", "cols2");
  const left = el("div"), right = el("div");
  const mkPanel = (title, findings, fallback) => {
    const p = el("div", "panel");
    p.appendChild(el("h3", null, title));
    if (!findings || !findings.length)
      p.appendChild(el("p", null,
        `<span style="color:var(--ink-muted)">${esc(fallback)}</span>`));
    else for (const f of findings) p.appendChild(renderFinding(f));
    return p;
  };
  left.appendChild(mkPanel("text &amp; encoding", [
    ...(pi.text_decoding || []), ...(pi.encoding_candidates || [])],
    "no reliable text encoding detected"));
  left.appendChild(mkPanel("structure findings",
    (pi.findings || []).filter((f) => f.category === "structure"),
    "no repeating structure detected"));
  left.appendChild(mkPanel("compression", pi.compression,
    "no compression signature"));
  left.appendChild(mkPanel("encryption assessment",
    pi.encryption_assessment ? [pi.encryption_assessment] : [],
    "not assessed"));
  right.appendChild(mkPanel("protocol candidates", pi.protocol_candidates,
    "no protocol validated"));
  const msgs = pi.messages || {};
  const pm = el("div", "panel");
  pm.appendChild(el("h3", null, "message reconstruction"));
  if (msgs.available) {
    pm.appendChild(el("p", null,
      `<span style="color:var(--ink2)">${esc(msgs.n_messages)} messages, ` +
      `${esc(msgs.header_bytes)} header byte(s) split from payload</span>`));
    const tbl = el("table", "kv");
    pm.appendChild(tbl);
    kvTable(tbl, (msgs.messages || []).slice(0, 12).map((m) => [
      m.index, m.length_bytes,
      html(`<span class="mono">${esc(m.header_hex || "")}</span>`),
      m.payload_text !== null && m.payload_text !== undefined
        ? m.payload_text
        : html(`<span class="mono">${esc(String(m.payload_hex || "").slice(0, 32))}...</span>`),
    ]), ["#", "bytes", "header", "payload"]);
  } else {
    pm.appendChild(el("p", null,
      `<span style="color:var(--ink-muted)">${esc(msgs.reason ||
        "no reliable message boundary reconstruction")}</span>`));
  }
  right.appendChild(pm);
  cols.appendChild(left); cols.appendChild(right);
  c.appendChild(cols);

  // limitations
  if ((pi.limitations || []).length) {
    const p = el("div", "panel");
    p.appendChild(el("h3", null, "limitations"));
    p.appendChild(el("ul", null,
      pi.limitations.map((l) => `<li style="color:var(--ink2);margin-left:16px">` +
        `${esc(l)}</li>`).join("")));
    c.appendChild(p);
  }

  // hexdump
  if (payloadHexStr) {
    const p = el("div", "panel");
    p.appendChild(el("h3", null, "payload hex"));
    const hd = el("div", "hexdump");
    p.appendChild(hd);
    const bytes = payloadHexStr.match(/.{2}/g).map((h) => parseInt(h, 16));
    hexdump(hd, bytes);
    c.appendChild(p);
  }
}

return { COL, $, el, esc, html, initTabs, badge, strengthBadge, meter, tile,
         kvTable, lineChart, barChart, scatterChart, heatmap, eyeChart,
         hexdump, renderPipeline, renderFinding, renderPayloadIntel };
})();

if (typeof module !== "undefined") module.exports = UI;
