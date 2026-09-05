"""D + H. Structure discovery and cross-frame field inference.

With multiple recovered frames (the normal S10 output) each byte column
is analysed across frames: constant columns are header/sync candidates,
monotonic columns are counters, columns correlating with message length
are length fields, high-variance regions are payload. Single-blob
payloads fall back to periodicity and entropy-transition analysis."""
from __future__ import annotations

import numpy as np

from .byte_forensics import shannon_entropy_bytes
from .models import Finding


def _column_matrix(frame_bytes: list) -> np.ndarray:
    width = min(len(f) for f in frame_bytes)
    return np.array([list(f[:width]) for f in frame_bytes], dtype=np.int64)


def infer_fields(frame_bytes: list) -> dict:
    """frame_bytes: list of per-frame byte strings (>= 4 frames)."""
    if len(frame_bytes) < 4:
        return {"available": False,
                "reason": "needs at least 4 frames for cross-frame statistics"}
    M = _column_matrix(frame_bytes)
    n_frames, width = M.shape
    lengths = np.array([len(f) for f in frame_bytes])
    cols = []
    for c in range(width):
        v = M[:, c]
        uniq = len(np.unique(v))
        diffs = np.diff(v)
        role, note = "payload/variable", ""
        if uniq == 1:
            role = "constant"
            note = f"always 0x{v[0]:02x}"
        elif uniq <= max(2, n_frames // 8) and (np.bincount(v, minlength=256).max()
                                                / n_frames) > 0.8:
            role = "mostly-constant/flag"
        elif np.all(diffs >= 0) and uniq > n_frames // 2:
            role = "counter/monotonic"
            note = f"increases by {int(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 0}"
        elif lengths.std() > 0 and abs(np.corrcoef(v, lengths)[0, 1]) > 0.9:
            role = "length-field candidate"
        # per-column entropy
        p = np.bincount(v, minlength=256) / n_frames
        pnz = p[p > 0]
        ent = float(-(pnz * np.log2(pnz)).sum()) / 8
        cols.append({"offset": c, "role": role, "entropy": round(ent, 3),
                     **({"note": note} if note else {})})
    # merge adjacent columns with the same role into fields
    fields = []
    start = 0
    for c in range(1, width + 1):
        if c == width or cols[c]["role"] != cols[start]["role"]:
            seg = cols[start:c]
            fields.append({
                "start_byte": start, "length_bytes": c - start,
                "role": seg[0]["role"],
                "mean_entropy": round(float(np.mean([x["entropy"] for x in seg])), 3),
                **({"note": seg[0].get("note", "")} if seg[0].get("note") else {}),
            })
            start = c
    return {"available": True, "n_frames": n_frames, "width_bytes": width,
            "columns": cols[:256], "fields": fields}


def detect_periodicity(data: bytes, max_period: int = 512) -> dict:
    """Byte-level record periodicity for single-blob payloads."""
    if len(data) < 64:
        return {"found": False}
    arr = np.frombuffer(data[:1 << 16], dtype=np.uint8).astype(np.float64)
    arr = arr - arr.mean()
    best_p, best_r = None, 0.0
    denom = float((arr * arr).mean()) + 1e-12
    for p in range(2, min(max_period, len(arr) // 4)):
        r = float(np.mean(arr[:-p] * arr[p:])) / denom
        if r > 0.3 and r > best_r:
            best_r, best_p = r, p
    if best_p is None:
        return {"found": False}
    return {"found": True, "period_bytes": best_p,
            "correlation": round(best_r, 3)}


def entropy_transitions(data: bytes, window: int = 32) -> list:
    """Offsets where the local entropy changes sharply (field boundaries)."""
    if len(data) < 4 * window:
        return []
    ents = []
    for off in range(0, len(data) - window, window):
        ents.append(shannon_entropy_bytes(data[off:off + window]) / 8)
    out = []
    for i in range(1, len(ents)):
        if abs(ents[i] - ents[i - 1]) > 0.35:
            out.append({"offset": i * window,
                        "from": round(ents[i - 1], 2),
                        "to": round(ents[i], 2)})
    return out[:32]


def discover_structure(data: bytes, frame_bytes: list = None) -> dict:
    result = {
        "periodicity": detect_periodicity(data),
        "entropy_transitions": entropy_transitions(data),
    }
    fi = infer_fields(frame_bytes or [])
    result["cross_frame"] = fi
    if fi.get("available"):
        result["fields"] = fi["fields"]
    findings = []
    if fi.get("available"):
        n_const = sum(1 for f in fi["fields"] if f["role"] == "constant")
        n_counter = sum(1 for f in fi["fields"] if f["role"] == "counter/monotonic")
        if n_const:
            findings.append(Finding(
                "structure", "constant header/marker region(s)",
                0.85, evidence=[f"{n_const} constant byte region(s) across "
                                f"{fi['n_frames']} frames"]))
        if n_counter:
            findings.append(Finding(
                "structure", "sequence counter candidate",
                0.8, evidence=["monotonically increasing byte column across "
                               "frames"]))
    elif result["periodicity"]["found"]:
        p = result["periodicity"]
        findings.append(Finding(
            "structure", f"repeating records every {p['period_bytes']} bytes",
            min(0.85, 0.5 + p["correlation"] / 2),
            evidence=[f"byte autocorrelation {p['correlation']} at lag "
                      f"{p['period_bytes']}"]))
    result["findings"] = findings
    return result
