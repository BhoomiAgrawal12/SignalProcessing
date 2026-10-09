"""I. Message reconstruction from S10 frame boundaries (or discovered
periodicity). Produces per-message offsets, header/payload split from
the field map, and decoded content previews."""
from __future__ import annotations

from .text_decoder import _text_quality


def reconstruct_messages(frame_bytes: list, structure: dict,
                         max_messages: int = 64) -> dict:
    if not frame_bytes:
        return {"available": False,
                "reason": "no frame boundaries recovered by S10"}
    fields = structure.get("fields") or []
    header_end = 0
    for f in fields:
        if f["role"] in ("constant", "mostly-constant/flag",
                         "counter/monotonic", "length-field candidate"):
            header_end = f["start_byte"] + f["length_bytes"]
        else:
            break
    messages = []
    offset = 0
    for i, fb in enumerate(frame_bytes[:max_messages]):
        body = fb[header_end:]
        text_q = _text_quality(body.decode("utf-8", errors="replace")) if body else 0
        messages.append({
            "index": i, "offset_bytes": offset, "length_bytes": len(fb),
            "header_hex": fb[:header_end].hex() if header_end else "",
            "payload_bytes": body,
            "payload_hex": body[:64].hex(),
            "payload_text": body.decode("utf-8", errors="replace")[:64]
            if text_q > 0.7 else None,
        })
        offset += len(fb)
    return {"available": True, "n_messages": len(frame_bytes),
            "header_bytes": header_end, "messages": messages}
