"""S11 orchestrator: runs the analyzer modules with resource limits,
combines their evidence into one classification, and preserves the
provenance of earlier stages (a payload without CRC validation is never
treated with CRC-validated certainty)."""
from __future__ import annotations

import time

import numpy as np

from .byte_forensics import analyze_bytes
from .compression_detector import detect_compression
from .encoding_detector import detect_encodings
from .encryption_assessor import assess_encryption
from .message_reconstruction import reconstruct_messages
from .models import Finding, confidence_band
from .protocol_fingerprint import fingerprint_protocols
from .structure_detector import discover_structure
from .text_decoder import try_text_decodings

VERSION = 1


def _frame_bytes_from_matrix(frames) -> list:
    """frames: (n, bits) 0/1 matrix from S10 -> per-frame byte strings."""
    if frames is None or len(frames) == 0:
        return []
    out = []
    for row in frames[:256]:
        n8 = (len(row) // 8) * 8
        if n8:
            out.append(np.packbits(np.asarray(row[:n8], dtype=np.uint8)).tobytes())
    return out


def analyze_payload(data: bytes, bits=None, frames=None,
                    provenance: dict = None, config=None) -> dict:
    """Main S11 entry point.

    data: recovered payload bytes; bits: payload bitstream; frames:
    stacked frame bit matrix from S10; provenance: upstream quality
    signals (crc_validated, fec_syndrome_rate, demod_locked...)."""
    t0 = time.perf_counter()
    provenance = provenance or {}
    max_bytes = getattr(config, "max_analysis_bytes", 1 << 20) if config else 1 << 20
    max_decomp = getattr(config, "max_decompression_bytes", 1 << 20) if config else 1 << 20
    data = (data or b"")[:max_bytes]

    if not data:
        return {"available": False, "reason": "no payload bytes recovered"}

    frame_bytes = _frame_bytes_from_matrix(frames)

    forensics = analyze_bytes(data, bits)
    text = try_text_decodings(data)
    encodings = detect_encodings(data)
    structure = discover_structure(data, frame_bytes)
    compression = detect_compression(data, max_output=max_decomp)
    messages = reconstruct_messages(frame_bytes, structure)
    protocols = fingerprint_protocols(data, messages.get("messages"))
    encryption = assess_encryption(data, text, compression, structure)

    # also fingerprint the per-message text bodies for framed payloads
    findings = []
    findings.extend(text[:2])
    findings.extend(encodings[:2])
    findings.extend(structure.get("findings", []))
    findings.extend(compression[:2])
    findings.extend(protocols[:3])
    findings.append(encryption)

    # provenance discounts: an unvalidated RF bit stream caps certainty.
    # Standalone byte analysis (empty provenance) is not capped.
    crc_ok = bool(provenance.get("crc_validated"))
    fec_rate = provenance.get("fec_syndrome_rate")
    cap = 1.0
    limitations = []
    if provenance and not crc_ok:
        cap = 0.8
        limitations.append("payload was recovered WITHOUT CRC validation; "
                           "all findings are capped accordingly")
    if fec_rate is not None and fec_rate < 0.9:
        cap = min(cap, 0.7)
        limitations.append(f"FEC syndrome-zero rate was {fec_rate:.2f}; "
                           "residual bit errors are possible")
    for f in findings:
        f.confidence = min(f.confidence, cap)

    # summary classification: the strongest content finding wins
    content = [f for f in findings if f.category in
               ("text", "protocol", "compression", "encoding")]
    if content and content[0].confidence >= 0.5:
        best = max(content, key=lambda f: f.confidence)
        summary = {"classification": best.verdict,
                   "confidence": round(best.confidence, 3),
                   "strength": confidence_band(best.confidence)}
    else:
        summary = {"classification": encryption.verdict,
                   "confidence": round(encryption.confidence, 3),
                   "strength": confidence_band(encryption.confidence)}

    # messages: strip raw bytes before serialisation
    msg_out = []
    for m in messages.get("messages", []):
        m = dict(m)
        m.pop("payload_bytes", None)
        msg_out.append(m)

    return {
        "available": True,
        "version": VERSION,
        "summary": summary,
        "byte_forensics": forensics,
        "text_decoding": [f.to_dict() for f in text],
        "encoding_candidates": [f.to_dict() for f in encodings],
        "structure": {k: v for k, v in structure.items() if k != "findings"},
        "compression": [f.to_dict() for f in compression],
        "encryption_assessment": encryption.to_dict(),
        "protocol_candidates": [f.to_dict() for f in protocols],
        "messages": {**{k: v for k, v in messages.items() if k != "messages"},
                     "messages": msg_out},
        "findings": [f.to_dict() for f in findings],
        "provenance": {"crc_validated": crc_ok,
                       "fec_syndrome_rate": fec_rate,
                       "demod_locked": provenance.get("demod_locked")},
        "limitations": limitations + [
            "S11 interprets recovered bytes; it never invents meaning - "
            "unknown is a valid result"],
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
