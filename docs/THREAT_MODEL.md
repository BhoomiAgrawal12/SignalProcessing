# Threat model

Dhwani analyses recordings it did not make. Treat every input as
attacker-controlled.

## Untrusted inputs and what handles them

| input | risk | handling | test |
|---|---|---|---|
| recording bytes (`.iq`, `.wav`) | malformed headers, huge files | defensive RIFF walk, memmap, first 2^22 samples analysed (warned) | `test_ingestion.py`, `test_conditioning.py` |
| file name / path | markup or YAML injection in exports | GRC path is a `json.dumps` scalar, numbers `float()`-coerced; PDF text through `xml.sax.saxutils.escape` | `test_reporting.py` |
| SigMF sidecar | wrong datatype / rate | parsed as JSON only; failures become a warning | `test_ingestion.py` |
| loaded `analysis.json` (web viewer, `dhwani report`) | DOM XSS, export injection | `UI.kvTable` / `UI.tile` escape by default, markup only via `UI.html()`; `report` re-render goes through the same escaped exporters | `web/test-node.js` |
| CVNet checkpoint | code execution via pickle | `torch.load(weights_only=True)` only; download pinned to a revision and SHA-256 | `test_cvnet_safety.py` |
| file name / format strings in CSV | spreadsheet formula injection | string cells starting with `= + - @` tab CR get a leading `'` | `test_reporting.py` |
| payload bytes | decompression bombs | bounded by `max_decompression_bytes` | `test_payload_intel.py` |

## Egress

The only outbound request is the pinned CVNet checkpoint download, made
when ML is enabled and no local checkpoint exists. Avoid it with
`--no-ml`, `HF_HUB_OFFLINE=1`, a local checkpoint, or the GUI's offline
box (checked by default when no local checkpoint is present). The web
pages read recordings through the File API and make one same-origin
request (`demo/analysis.json`).

## Data at rest

- Payload bytes and `recording.file_path` appear in every export (JSON,
  `_payload.bin`, `.hex.txt`, PDF, web hexdump) and in the signature
  database. Nothing is redacted. Any future redaction must cover all of
  these, or none.
- The S2 cache pickles detection results into `~/.cache/dhwani`. It is
  written and read only by Dhwani; a cache directory shared with an
  untrusted user would be a code-execution path, so keep it private.

## Notes

- The `.grc` flowgraph is a starting point the analyst runs in GNU
  Radio; it is never executed here.
