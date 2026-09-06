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


def extract_sync_word(frames: np.ndarray, entropy: np.ndarray,
                      sync_thr: float = 0.15) -> dict:
    """The longest run of near-constant columns is the sync word."""
    const = entropy < sync_thr
    best_start, best_len = 0, 0
    i = 0
    n = len(const)
    while i < n:
        if const[i]:
            j = i
            while j < n and const[j]:
                j += 1
            if j - i > best_len:
                best_start, best_len = i, j - i
            i = j
        else:
            i += 1
    if best_len < 8:
        return {"found": False}
    maj = (frames[:, best_start:best_start + best_len].mean(axis=0) > 0.5).astype(np.uint8)
    pad = (-len(maj)) % 8
    padded = np.concatenate([maj, np.zeros(pad, dtype=np.uint8)])
    word_hex = np.packbits(padded).tobytes().hex()
    return {"found": True, "offset_bits": int(best_start),
            "length_bits": int(best_len), "hex": word_hex,
            "bits": maj}


def analyze_frames(bits: np.ndarray, frame_length: int) -> dict:
    """Stack frames, compute entropy map, sync word, field map."""
    n_rows = len(bits) // frame_length
    frames = np.asarray(bits[: n_rows * frame_length],
                        dtype=np.uint8).reshape(n_rows, frame_length)
    ent = column_entropy(frames)
    sync = extract_sync_word(frames, ent)
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
