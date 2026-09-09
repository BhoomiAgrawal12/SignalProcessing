"""Frame-period detection, sync-word extraction, per-column entropy field
mapping and payload statistics (stage S10)."""
from __future__ import annotations

import math

import numpy as np

from ..bits.packing import pack_bits, popcount_u64


def binary_autocorrelation(bits: np.ndarray, max_lag: int,
                           max_bits: int = 500000) -> np.ndarray:
    """ac[tau] = fraction of positions where b[i] == b[i+tau].

    Bit-packed XOR + popcount: millions of bits in milliseconds.
    Random data -> 0.5; peaks mark the frame period."""
    bits = np.asarray(bits[:max_bits], dtype=np.uint8)
    n = len(bits)
    packed = pack_bits(bits)
    out = np.zeros(max_lag + 1)
    out[0] = 1.0
    for tau in range(1, max_lag + 1):
        m = n - tau
        if m < 64:
            break
        # shift by tau: whole-word part + bit part
        a = bits[:m]
        b = bits[tau:tau + m]
        pa, pb = pack_bits(a), pack_bits(b)
        diff = popcount_u64(pa ^ pb).sum()
        # subtract padding contribution: pad bits are zero in both -> equal
        out[tau] = 1.0 - diff / m
    return out


def find_frame_length(bits: np.ndarray, min_len: int = 16,
                      max_len: int = 8192, top_k: int = 5) -> list:
    """Combine autocorrelation peaks and column-stability scores.

    Returns candidates [{'length', 'score', 'autocorr', 'stability'}...],
    best first, fundamental periods preferred over their multiples."""
    bits = np.asarray(bits, dtype=np.uint8)
    max_len = min(max_len, len(bits) // 8)
    if max_len < min_len:
        return []
    ac = binary_autocorrelation(bits, max_len)
    # column stability for promising lags only (cheap pre-filter via ac)
    base = np.median(ac[min_len:])
    dev = ac - base
    lags = np.argsort(dev[min_len:max_len + 1])[::-1][:64] + min_len
    cands = []
    for P in lags:
        P = int(P)
        n_rows = len(bits) // P
        if n_rows < 4:
            continue
        m = bits[: n_rows * P].reshape(n_rows, P)
        col_mean = m.mean(axis=0)
        # subtract the expected |mean-0.5| of random data (~0.399/sqrt(n))
        # so that few-row (large-P) trials are not spuriously favoured
        raw = float(np.abs(col_mean - 0.5).mean()) * 2
        stability = max(0.0, raw - 0.7979 / np.sqrt(n_rows))
        score = float(dev[P]) + stability
        # real protocols are overwhelmingly byte-aligned: prefer multiples
        # of 8 when scores are close (also what the CRC hunter needs)
        if P % 8 == 0:
            score *= 1.08
        cands.append({"length": P, "score": score,
                      "autocorr": float(ac[P]), "stability": stability})
    cands.sort(key=lambda c: -c["score"])
    if not cands:
        return []

    def stability_of(P):
        n_rows = len(bits) // P
        if n_rows < 4:
            return 0.0
        m = bits[: n_rows * P].reshape(n_rows, P)
        raw = float(np.abs(m.mean(axis=0) - 0.5).mean()) * 2
        return max(0.0, raw - 0.7979 / np.sqrt(n_rows))

    # Prefer the fundamental period: a multiple of the true frame length
    # scores just as well on autocorrelation/stability, so test the
    # divisors of the winner and take the smallest one that explains the
    # structure equally well.
    best = cands[0]
    P = best["length"]
    for d in sorted(d for d in range(min_len, P) if P % d == 0):
        stab_d = stability_of(d)
        if stab_d >= 0.92 * best["stability"]:
            cands.insert(0, {"length": d, "score": best["score"] + 0.01,
                             "autocorr": float(ac[d]) if d < len(ac) else 0.0,
                             "stability": stab_d})
            break
    seen, kept = set(), []
    for c in cands:
        if c["length"] in seen:
            continue
        seen.add(c["length"])
        kept.append(c)
        if len(kept) >= top_k:
            break
    return kept


def column_entropy(frames: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of each bit position across stacked frames."""
    p1 = frames.mean(axis=0)
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    return (-p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1))


def classify_fields(entropy: np.ndarray, sync_thr: float = 0.15,
                    payload_thr: float = 0.85) -> list:
    """Segment bit positions into contiguous fields by entropy level."""
    def label(e):
        if e < sync_thr:
            return "sync/constant"
        if e > payload_thr:
            return "payload/random"
        return "header/varying"
    fields = []
    start = 0
    cur = label(entropy[0])
    for i in range(1, len(entropy)):
        lab = label(entropy[i])
        if lab != cur:
            fields.append({"start_bit": start, "length_bits": i - start,
                           "role": cur,
                           "mean_entropy": round(float(entropy[start:i].mean()), 3)})
            start, cur = i, lab
    fields.append({"start_bit": start, "length_bits": len(entropy) - start,
                   "role": cur,
                   "mean_entropy": round(float(entropy[start:].mean()), 3)})
    return fields


def _trim_sync_word(bits: np.ndarray, signature_words: list = None) -> tuple:
    """Trim a constant-column run down to the sync word it contains.

    Report §5.2: the run is rounded down to a byte boundary but its
    trailing constant bytes are never removed, so a 24-bit constant run
    turns ``eb90`` into ``eb9000``.  The problem appeared in all ten
    operator reports, and an over-long sync word is not a cosmetic
    defect - it is what an analyst matches against a signature library
    and feeds to a receiver.

    Two rules, in order of authority:

    1. If a prefix of the run is a word the signature library already
       knows, that prefix is the sync word.  A library match is direct
       evidence; nothing else here is.
    2. Otherwise drop trailing all-zero or all-one bytes.  Those are the
       constant leading bytes of whatever field follows - a zeroed
       length, a fixed protocol version, the high bits of a small
       counter - and a real sync word is chosen for its correlation
       properties, which all-zero or all-one bytes destroy.

    At least two bytes are always kept: a one-byte "sync word" is not
    evidence of anything, and trimming to it would be worse than the
    over-long word it replaced.  Returns (trimmed bits, reason).
    """
    n_bytes = len(bits) // 8
    if n_bytes < 2:
        return bits, ""
    whole = bits[:n_bytes * 8].reshape(n_bytes, 8)
    if signature_words:
        as_hex = np.packbits(bits[:n_bytes * 8]).tobytes().hex()
        for k in range(n_bytes, 1, -1):
            prefix = as_hex[:2 * k]
            if any(str(w).lower().lstrip("0x") == prefix
                   for w in signature_words):
                if k < n_bytes:
                    return (bits[:k * 8],
                            f"trimmed to the longest prefix present in the "
                            f"signature library ({prefix})")
                return bits, ""
    keep = n_bytes
    while keep > 2:
        last = whole[keep - 1]
        if last.all() or not last.any():
            keep -= 1
        else:
            break
    if keep == n_bytes:
        return bits, ""
    dropped = n_bytes - keep
    return (bits[:keep * 8],
            f"dropped {dropped} trailing all-zero/all-one byte"
            f"{'s' if dropped != 1 else ''}: constant bytes of the field "
            "that follows the sync word, not part of it")


def extract_sync_word(frames: np.ndarray, entropy: np.ndarray,
                      sync_thr: float = 0.15,
                      signature_words: list = None) -> dict:
    """The near-constant column run anchored at the frame start is the
    sync word. The frame alignment upstream is already fixed by rank and
    CRC evidence, so a real sync word sits at bit 0; structured payloads
    (constant text characters) produce interior constant runs that can
    be LONGER than the sync word and must not steal the label.

    The reported word is the constant run trimmed by
    :func:`_trim_sync_word`; ``length_bits`` still describes the FULL
    constant run, because that is where the payload starts regardless of
    how much of the run is really the sync word.
    """
    const = entropy < sync_thr
    runs = []
    i, n = 0, len(const)
    while i < n:
        if const[i]:
            j = i
            while j < n and const[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    anchored = [r for r in runs if r[0] == 0 and r[1] >= 8]
    if anchored:
        best_start, best_len = anchored[0]
    else:
        best_start, best_len = max(runs, key=lambda r: r[1],
                                   default=(0, 0))
    if best_len < 8:
        return {"found": False}
    # report whole bytes: a constant run often extends a few bits past
    # the true word (constant MSB of a following counter field)
    word_len = (best_len // 8) * 8 if best_len >= 16 else best_len
    maj = (frames[:, best_start:best_start + best_len].mean(axis=0)
           > 0.5).astype(np.uint8)
    word = maj[:word_len] if word_len else maj
    word, trim_reason = _trim_sync_word(word, signature_words)
    pad = (-len(word)) % 8
    padded = np.concatenate([word, np.zeros(pad, dtype=np.uint8)])
    word_hex = np.packbits(padded).tobytes().hex()
    out = {"found": True, "offset_bits": int(best_start),
           "length_bits": int(best_len), "word_bits": int(len(word)),
           "constant_run_bits": int(best_len),
           "hex": word_hex, "bits": maj}
    if trim_reason:
        out["trimmed"] = trim_reason
    return out


def analyze_frames(bits: np.ndarray, frame_length: int,
                   signature_words: list = None) -> dict:
    """Stack frames, compute entropy map, sync word, field map.

    ``signature_words`` are known sync words from the signature library;
    when the detected constant run starts with one of them, that prefix
    is the sync word (see :func:`_trim_sync_word`)."""
    n_rows = len(bits) // frame_length
    frames = np.asarray(bits[: n_rows * frame_length],
                        dtype=np.uint8).reshape(n_rows, frame_length)
    ent = column_entropy(frames)
    sync = extract_sync_word(frames, ent, signature_words=signature_words)
    fields = classify_fields(ent)
    return {"frames": frames, "entropy": ent, "sync": sync, "fields": fields}


def payload_stats(data: bytes) -> dict:
    """Byte-level entropy and printability - the honest 'is it encrypted?'
    answer (never claim decryption; report entropy near 1.0 bit/bit)."""
    if not data:
        return {"entropy_bits_per_bit": None, "printable_fraction": None,
                "likely_encrypted": False}
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(float)
    p = counts / counts.sum()
    nz = p > 0
    H = float(-(p[nz] * np.log2(p[nz])).sum()) / 8.0   # bits per bit
    printable = float(((arr >= 0x20) & (arr < 0x7F)).mean())
    return {"entropy_bits_per_bit": round(H, 4),
            "printable_fraction": round(printable, 4),
            "likely_encrypted": bool(H > 0.95 and printable < 0.5)}
