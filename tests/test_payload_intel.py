"""S11 payload intelligence: the spec's required cases."""
import gzip
import struct

import numpy as np
import pytest

from rfanalyzer.payload import analyze_payload
from rfanalyzer.payload.models import confidence_band


def test_plain_text():
    r = analyze_payload(b"HELLO WORLD this is a plain transmission. " * 4)
    assert r["available"]
    assert r["text_decoding"][0]["confidence"] > 0.8
    assert "text" in r["summary"]["classification"].lower()


def test_json_detected():
    r = analyze_payload(b'{"temperature":25,"status":"OK"}')
    assert any(p["verdict"] == "JSON document" and p["confidence"] > 0.9
               for p in r["protocol_candidates"])


def test_base64_candidate():
    r = analyze_payload(b"SGVsbG8gV29ybGQsIGJhc2U2NCB3cmFwcGVkIG1lc3NhZ2Uh")
    hit = [e for e in r["encoding_candidates"] if "Base64" in e["verdict"]]
    assert hit and "Hello World" in hit[0]["details"]["decoded_preview"]


def test_gzip_detected_and_decompressed():
    blob = gzip.compress(b"the quick brown fox " * 50)
    r = analyze_payload(blob)
    hit = [c for c in r["compression"] if c["details"].get("decompressed")]
    assert hit and hit[0]["confidence"] > 0.9


def test_random_not_called_encrypted():
    rng = np.random.default_rng(0)
    r = analyze_payload(bytes(rng.integers(0, 256, 4096, dtype=np.uint8)))
    ea = r["encryption_assessment"]
    assert ea["verdict"] in ("possibly encrypted or compressed",
                             "insufficient evidence")
    assert any("cannot distinguish" in l for l in ea["limitations"])


def test_structured_records(rng):
    frames = []
    for i in range(24):
        body = b"\xEB\x90" + bytes([i, 12]) + \
            bytes(rng.integers(0, 256, 8, dtype=np.uint8))
        frames.append(np.unpackbits(np.frombuffer(body, dtype=np.uint8)))
    frames = np.array(frames)
    data = b"".join(np.packbits(f).tobytes() for f in frames)
    r = analyze_payload(data, frames=frames,
                        provenance={"crc_validated": True})
    roles = [f["role"] for f in r["structure"]["fields"]]
    assert "constant" in roles
    assert "counter/monotonic" in roles
    assert r["messages"]["available"]
    assert r["messages"]["n_messages"] == 24


def test_provenance_caps_confidence():
    r = analyze_payload(b"HELLO WORLD " * 20,
                        provenance={"crc_validated": False})
    assert all(f["confidence"] <= 0.8 for f in r["findings"])
    assert any("without CRC" in l or "WITHOUT CRC" in l
               for l in r["limitations"])
    assert r["summary"].get("trust") in ("SPECULATIVE PAYLOAD",
                                         "PROBABLE PAYLOAD")


def test_ipv4_checksum_validation():
    hdr = bytearray(20)
    hdr[0] = 0x45
    struct.pack_into(">H", hdr, 2, 28)
    hdr[8], hdr[9] = 64, 17
    hdr[12:16] = bytes([192, 168, 1, 10])
    hdr[16:20] = bytes([10, 0, 0, 1])
    s = 0
    for i in range(0, 20, 2):
        s += (hdr[i] << 8) + hdr[i + 1]
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    struct.pack_into(">H", hdr, 10, (~s) & 0xFFFF)
    r = analyze_payload(bytes(hdr) + bytes(8))
    hit = [p for p in r["protocol_candidates"] if p["verdict"] == "IPv4 packet"]
    assert hit and hit[0]["confidence"] >= 0.9


def test_utf16_prior():
    real = "TEMPERATURE 25 OK ".encode("utf-16-le") * 4
    r = analyze_payload(real)
    assert "UTF-16" in r["text_decoding"][0]["verdict"]
    # random printable bytes must not be claimed as UTF-16
    rng = np.random.default_rng(3)
    junk = bytes(rng.integers(0x20, 0x7F, 200, dtype=np.uint8))
    r2 = analyze_payload(junk)
    top = r2["text_decoding"][0]["verdict"] if r2["text_decoding"] else ""
    assert not top.startswith("UTF-16")


def test_confidence_bands():
    assert confidence_band(0.95) == "VALIDATED"
    assert confidence_band(0.8) == "LIKELY"
    assert confidence_band(0.6) == "POSSIBLE"
    assert confidence_band(0.3) == "WEAK"


def test_empty_payload():
    r = analyze_payload(b"")
    assert not r["available"]
