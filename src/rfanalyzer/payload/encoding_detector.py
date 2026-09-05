"""C. Encoding discovery: is the payload itself an encoded wrapper
(Base64, hex ASCII, URL encoding)? A claim requires a successful decode
plus the decoded bytes showing MORE structure than the original."""
from __future__ import annotations

import base64
import binascii
import re
import urllib.parse

from .byte_forensics import shannon_entropy_bytes
from .models import Finding
from .text_decoder import _text_quality


def _structure_gain(before: bytes, after: bytes) -> float:
    """Positive when decoding revealed structure (lower entropy or text)."""
    if not after:
        return -1.0
    e_before = shannon_entropy_bytes(before) / 8
    e_after = shannon_entropy_bytes(after) / 8
    try:
        text_after = _text_quality(after.decode("utf-8", errors="replace"))
    except Exception:
        text_after = 0.0
    return (e_before - e_after) + max(0.0, text_after - 0.5)


def detect_encodings(data: bytes, max_bytes: int = 1 << 16) -> list:
    data = data[:max_bytes]
    findings = []
    stripped = data.strip()

    # Base64
    if len(stripped) >= 8 and len(stripped) % 4 == 0 and \
            re.fullmatch(rb"[A-Za-z0-9+/]+={0,2}", stripped):
        try:
            decoded = base64.b64decode(stripped, validate=True)
            gain = _structure_gain(stripped, decoded)
            if gain > 0.1:
                findings.append(Finding(
                    category="encoding", verdict="Base64 wrapper",
                    confidence=min(0.95, 0.7 + gain / 2),
                    evidence=["valid Base64 alphabet and padding",
                              "successful strict decode",
                              f"decoded data shows more structure "
                              f"(gain {gain:.2f})"],
                    details={"decoded_preview": decoded[:120].decode(
                        "utf-8", errors="replace"),
                        "decoded_bytes": len(decoded)}))
        except (binascii.Error, ValueError):
            pass

    # hex ASCII
    if len(stripped) >= 16 and len(stripped) % 2 == 0 and \
            re.fullmatch(rb"[0-9a-fA-F\s]+", stripped):
        try:
            decoded = binascii.unhexlify(re.sub(rb"\s", b"", stripped))
            gain = _structure_gain(stripped, decoded)
            if gain > 0.1:
                findings.append(Finding(
                    category="encoding", verdict="hex-encoded ASCII wrapper",
                    confidence=min(0.9, 0.65 + gain / 2),
                    evidence=["all characters are hex digits",
                              f"decode reveals structure (gain {gain:.2f})"],
                    details={"decoded_preview": decoded[:120].decode(
                        "utf-8", errors="replace")}))
        except (binascii.Error, ValueError):
            pass

    # URL encoding
    if b"%" in data and re.search(rb"%[0-9a-fA-F]{2}", data):
        try:
            text = data.decode("ascii")
            decoded = urllib.parse.unquote(text).encode()
            n_esc = len(re.findall(r"%[0-9a-fA-F]{2}", text))
            if decoded != data and n_esc >= 3 and \
                    _text_quality(decoded.decode("utf-8", errors="replace")) > 0.8:
                findings.append(Finding(
                    category="encoding", verdict="URL-encoded text",
                    confidence=0.8,
                    evidence=[f"{n_esc} percent-escapes decode to clean text"],
                    details={"decoded_preview":
                             decoded[:120].decode("utf-8", errors="replace")}))
        except UnicodeDecodeError:
            pass
    findings.sort(key=lambda f: -f.confidence)
    return findings
