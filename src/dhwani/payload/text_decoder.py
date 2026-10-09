"""B. Conservative text decoding: report an encoding only when the
decode is valid AND the result actually looks like text."""
from __future__ import annotations

from .models import Finding


def _text_quality(s: str) -> float:
    if not s:
        return 0.0
    printable = sum(1 for c in s if c.isprintable() or c in "\r\n\t")
    letters = sum(1 for c in s if c.isalpha() or c.isspace() or c.isdigit())
    return (printable / len(s)) * 0.5 + (letters / len(s)) * 0.5


def try_text_decodings(data: bytes, max_bytes: int = 1 << 16) -> list:
    data = data[:max_bytes]
    findings = []
    for name, codec in (("ASCII", "ascii"), ("UTF-8", "utf-8"),
                        ("UTF-16-LE", "utf-16-le"), ("UTF-16-BE", "utf-16-be"),
                        ("Latin-1", "latin-1")):
        try:
            decoded = data.decode(codec)
            errors = 0
        except UnicodeDecodeError:
            decoded = data.decode(codec, errors="replace")
            errors = decoded.count("�")
        if not decoded:
            continue
        validity = 1.0 - errors / max(1, len(decoded))
        quality = _text_quality(decoded)
        score = validity * quality
        # Latin-1 always "decodes"; demand real text quality from it
        if name == "Latin-1" and quality < 0.85:
            continue
        # UTF-16 of Latin-script text carries ~50% null bytes; arbitrary
        # byte pairs decode to "printable" CJK and would spuriously score
        # high, so demand one of those signatures before trusting UTF-16
        if name.startswith("UTF-16"):
            null_ratio = data.count(0) / max(1, len(data))
            ascii_ratio = sum(1 for ch in decoded if ord(ch) < 0x250) / \
                max(1, len(decoded))
            if null_ratio < 0.2 and ascii_ratio < 0.5:
                score *= 0.3
        if score < 0.6:
            continue
        findings.append(Finding(
            category="text",
            verdict=f"{name} text",
            confidence=min(0.98, score),
            evidence=[f"decode validity {validity:.2f}",
                      f"text quality {quality:.2f}"],
            limitations=["preview truncated"] if len(data) == max_bytes else [],
            details={"encoding": name,
                     "preview": decoded[:200],
                     "printable_ratio": round(quality, 3),
                     "replacement_count": errors}))
    findings.sort(key=lambda f: -f.confidence)
    return findings
