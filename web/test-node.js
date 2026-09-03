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

const FIX = path.join(__dirname, "..", "examples", "webtest");
const cases = [
  ["qpsk_c64.iq", "QPSK", {}],
  ["fsk2_int16.iq", "2FSK", {}],
  ["qam16.wav", "16QAM", {}],
  ["bpsk_u8.iq", "BPSK", {}],
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
