"""E. Compression detection: signature + bounded decompression + output
plausibility. A magic-byte match alone never produces a claim."""
from __future__ import annotations

import bz2
import lzma
import zlib

from .byte_forensics import shannon_entropy_bytes
from .models import Finding

_SIGNATURES = [
    ("gzip", b"\x1f\x8b", lambda d, lim: zlib.decompressobj(31).decompress(d, lim)),
    ("zlib", b"\x78", lambda d, lim: zlib.decompressobj(15).decompress(d, lim)),
    ("bzip2", b"BZh", lambda d, lim: bz2.BZ2Decompressor().decompress(d, lim)),
    ("xz", b"\xfd7zXZ\x00", lambda d, lim: lzma.LZMADecompressor().decompress(d, lim)),
    ("zip", b"PK\x03\x04", None),
    ("zstd", b"\x28\xb5\x2f\xfd", None),
    ("lz4", b"\x04\x22\x4d\x18", None),
]


def detect_compression(data: bytes, max_output: int = 1 << 20,
                       max_input: int = 1 << 20) -> list:
    data = data[:max_input]
    findings = []
    for name, magic, decomp in _SIGNATURES:
        for off in (0,):    # payloads from S10 are already frame-aligned
            if not data[off:].startswith(magic):
                continue
            if name == "zlib" and (len(data) < 2 or
                                   (data[0] * 256 + data[1]) % 31 != 0):
                continue    # zlib header checksum rule (FCHECK)
            if decomp is None:
                findings.append(Finding(
                    category="compression",
                    verdict=f"{name} container signature",
                    confidence=0.55,
                    evidence=[f"magic bytes {magic.hex()} at offset {off}"],
                    limitations=[f"no bounded {name} decompressor in the "
                                 "standard library; signature only"],
                    details={"format": name, "decompressed": False}))
                continue
            try:
                out = decomp(data[off:], max_output)
            except Exception as e:
                findings.append(Finding(
                    category="compression",
                    verdict=f"{name} signature but decompression failed",
                    confidence=0.3,
                    evidence=[f"magic bytes matched, error: {e}"],
                    limitations=["signature may be coincidental"],
                    details={"format": name, "decompressed": False}))
                continue
            if len(out) == 0:
                continue
            ent = shannon_entropy_bytes(out) / 8
            findings.append(Finding(
                category="compression",
                verdict=f"{name} compressed data",
                confidence=0.95,
                evidence=[f"magic bytes {magic.hex()}",
                          "bounded decompression succeeded",
                          f"{len(out)} bytes out, entropy {ent:.2f}"],
                limitations=(["output truncated at limit"]
                             if len(out) >= max_output else []),
                details={"format": name, "decompressed": True,
                         "output_bytes": len(out),
                         "output_entropy": round(ent, 3),
                         "preview": out[:120].decode("utf-8", errors="replace")}))
    findings.sort(key=lambda f: -f.confidence)
    return findings
