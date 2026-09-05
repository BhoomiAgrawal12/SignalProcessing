"""F. Conservative encryption assessment.

High entropy alone cannot distinguish encryption from compression; that
limitation is attached to every high-entropy assessment, always."""
from __future__ import annotations

import zlib

from .byte_forensics import shannon_entropy_bytes
from .models import Finding

_LIMIT = ("high entropy alone cannot distinguish encryption from "
          "compression or random data")


def assess_encryption(data: bytes, text_findings: list,
                      compression_findings: list,
                      structure: dict) -> Finding:
    if not data:
        return Finding("encryption", "no payload", 0.0)
    ent = shannon_entropy_bytes(data[:1 << 20]) / 8
    # actual compressibility: how much does deflate shrink it?
    sample = data[:1 << 16]
    ratio = len(zlib.compress(sample, 6)) / max(1, len(sample))

    if any(f.confidence > 0.8 for f in text_findings):
        return Finding("encryption", "likely plaintext", 0.9,
                       evidence=["high-confidence text decoding succeeded"],
                       details={"entropy": round(ent, 3)})
    if any(f.confidence > 0.9 for f in compression_findings):
        return Finding("encryption", "compressed (validated)", 0.9,
                       evidence=["compression signature plus successful "
                                 "bounded decompression"],
                       details={"entropy": round(ent, 3)})
    if ent < 0.7 or ratio < 0.8:
        return Finding("encryption", "likely structured binary", 0.75,
                       evidence=[f"entropy {ent:.2f} bits/bit",
                                 f"deflate compresses to {ratio:.2f} of "
                                 "original (redundancy present)"],
                       details={"entropy": round(ent, 3),
                                "compress_ratio": round(ratio, 3)})
    if ent > 0.93 and ratio > 0.98:
        low_structure = not structure or not structure.get("fields")
        return Finding(
            "encryption", "possibly encrypted or compressed",
            0.62 if low_structure else 0.5,
            evidence=[f"entropy {ent:.2f} bits/bit",
                      "near-uniform byte distribution",
                      f"incompressible (deflate ratio {ratio:.2f})",
                      "no recognised compression signature"],
            limitations=[_LIMIT],
            details={"entropy": round(ent, 3),
                     "compress_ratio": round(ratio, 3)})
    return Finding("encryption", "insufficient evidence", 0.4,
                   evidence=[f"entropy {ent:.2f} bits/bit",
                             f"deflate ratio {ratio:.2f}"],
                   limitations=[_LIMIT],
                   details={"entropy": round(ent, 3),
                            "compress_ratio": round(ratio, 3)})
