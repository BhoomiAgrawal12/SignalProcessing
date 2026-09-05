"""G. Protocol fingerprinting with validation, not pattern matching.

Detector registry: each detector returns None or a Finding whose
confidence reflects how much of the structure actually validated."""
from __future__ import annotations

import json
import re
import struct

from .models import Finding


def _detect_json(data: bytes) -> Finding | None:
    s = data.strip()
    if not s[:1] in (b"{", b"["):
        return None
    try:
        obj = json.loads(s.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        # try prefix parse: payload may contain trailing bytes
        try:
            dec = json.JSONDecoder()
            obj, end = dec.raw_decode(s.decode("utf-8", errors="replace"))
            if end < len(s) * 0.5:
                return None
        except ValueError:
            return None
    kind = type(obj).__name__
    return Finding("protocol", "JSON document", 0.97,
                   evidence=["parses as strict JSON",
                             f"top-level {kind}"],
                   details={"top_level_type": kind,
                            "keys": list(obj.keys())[:10]
                            if isinstance(obj, dict) else None})


def _detect_http(data: bytes) -> Finding | None:
    head = data[:512]
    m = re.match(rb"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) \S+ HTTP/1\.[01]\r\n",
                 head) or re.match(rb"HTTP/1\.[01] \d{3} ", head)
    if not m:
        return None
    n_headers = head.count(b"\r\n")
    return Finding("protocol", "HTTP/1.x message", 0.92,
                   evidence=["valid request/status line",
                             f"{n_headers} CRLF-terminated lines"],
                   details={"start_line": head.split(b"\r\n")[0]
                            .decode("latin-1")[:80]})


def _ipv4_checksum(header: bytes) -> int:
    s = 0
    for i in range(0, len(header), 2):
        s += (header[i] << 8) + (header[i + 1] if i + 1 < len(header) else 0)
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def _detect_ipv4(data: bytes) -> Finding | None:
    if len(data) < 20:
        return None
    v_ihl = data[0]
    if v_ihl >> 4 != 4:
        return None
    ihl = (v_ihl & 0xF) * 4
    if ihl < 20 or len(data) < ihl:
        return None
    total_len = struct.unpack(">H", data[2:4])[0]
    if total_len < ihl or total_len > len(data) + 64:
        return None
    checksum_ok = _ipv4_checksum(data[:ihl]) == 0
    proto = data[9]
    conf = 0.9 if checksum_ok else 0.55
    ev = ["version nibble = 4", f"header length {ihl} valid",
          f"total length {total_len} plausible"]
    ev.append("header checksum valid" if checksum_ok
              else "header checksum INVALID")
    return Finding("protocol", "IPv4 packet", conf, evidence=ev,
                   limitations=[] if checksum_ok else
                   ["checksum failed: could be a coincidental match"],
                   details={"protocol_number": proto,
                            "src": ".".join(map(str, data[12:16])),
                            "dst": ".".join(map(str, data[16:20]))})


_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"%PDF-", "PDF document"),
    (b"\x7fELF", "ELF binary"),
    (b"MZ", "DOS/PE executable"),
    (b"RIFF", "RIFF container"),
    (b"OggS", "Ogg container"),
]


def _detect_magic(data: bytes) -> Finding | None:
    for magic, name in _MAGIC:
        if data.startswith(magic) and len(data) > len(magic) + 8:
            return Finding("protocol", f"{name} signature", 0.7,
                           evidence=[f"magic bytes {magic.hex()} at offset 0"],
                           limitations=["signature only; content not parsed"],
                           details={"format": name})
    return None


DETECTORS = [_detect_json, _detect_http, _detect_ipv4, _detect_magic]


def fingerprint_protocols(data: bytes, messages: list = None,
                          max_bytes: int = 1 << 16) -> list:
    """Run every detector on the whole payload and on the first message
    payloads when message boundaries are known."""
    findings = []
    targets = [data[:max_bytes]]
    for msg in (messages or [])[:4]:
        body = msg.get("payload_bytes")
        if body and len(body) >= 8:
            targets.append(body[:max_bytes])
    seen = set()
    for t in targets:
        for det in DETECTORS:
            try:
                f = det(t)
            except Exception:
                continue
            if f and f.verdict not in seen:
                seen.add(f.verdict)
                findings.append(f)
    findings.sort(key=lambda f: -f.confidence)
    return findings
