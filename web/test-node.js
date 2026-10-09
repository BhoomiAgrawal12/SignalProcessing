/* Node regression harness for the browser analysis engine.
 *
 * Generate the fixtures first (needs the Python package):
 *   .venv/bin/python web/make-test-fixtures.py
 * Then:
 *   node web/test-node.js
 */
"use strict";
const fs = require("fs");
const path = require("path");
const A = require("./analyze.js");

/* 1.D: values from a loaded analysis.json are rendered inert. Minimal DOM
 * stub: a cell's markup only reaches .innerHTML when wrapped in UI.html(). */
global.document = { createElement: (tag) => ({
  tag, kids: [], textContent: "", html: null,
  appendChild(c) { this.kids.push(c); return c; },
  set innerHTML(v) { this.html = v; this.kids = []; },
  get innerHTML() { return this.html || ""; } }) };
const UI = require("./ui.js");
const evil = '<img src=x onerror=alert(1)>';
const t = UI.kvTable(document.createElement("table"),
  [[evil, UI.html("<b>ok</b>"), null]]);
const [c0, c1, c2] = t.kids[0].kids;
const domOk = c0.html === null && c0.textContent === evil &&
  c1.html === "<b>ok</b>" && c2.textContent === "\u2013" &&
  !UI.tile(evil, evil, evil).includes("<img") &&
  !UI.meter(evil).includes("<img");
console.log(`kvTable/tile/meter escape untrusted values: ${domOk ? "PASS" : "FAIL"}`);
if (!domOk) process.exit(1);

const FIX = path.join(__dirname, "..", "examples", "webtest");
const cases = [
  ["qpsk_c64.iq", "QPSK", {}],
  ["fsk2_int16.iq", "2FSK", {}],
  ["qam16.wav", "16QAM", {}],
  ["bpsk_u8.iq", "BPSK", {}],
  ["fsk2_mono.wav", "2FSK", {}],
  ["fsk2_4sps.iq", "2FSK", {}],
  ["../../web/demo/demo_chain.wav", "QPSK", {}],   // the shared demo (scripts/make_demo_report.py)
];

let pass = 0;
for (const [name, expected, ov] of cases) {
  const p = path.join(FIX, name);
  if (!fs.existsSync(p)) {
    console.log(`SKIP ${name} (fixture missing; run make-test-fixtures.py)`);
    continue;
  }
  const buf = fs.readFileSync(p);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const r = A.analyzeRecording(ab, name, ov);
  const got = r.modulation ? r.modulation.prediction : "none";
  const ok = got === expected;
  pass += ok;
  console.log(`${name.padEnd(16)} dtype=${(r.recording.datatype || "").padEnd(14)} ` +
    `Rs=${r.parameters ? r.parameters.symbol_rate_norm : "-"} -> ${got} ` +
    (ok ? "PASS" : `FAIL (expected ${expected})`));
}
console.log(`${pass}/${cases.length} passed`);
process.exit(pass === cases.length ? 0 : 1);
