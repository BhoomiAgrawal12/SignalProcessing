"""Berlekamp-Massey over GF(2) and additive-scrambler detection.

BM recovers the shortest LFSR generating a binary sequence.  For an
additive scrambler XORed onto *structured* data we cannot run BM on the raw
stream directly, but two practical situations are handled:

1. Idle/preamble stretches where the underlying data is constant: the
   stream section IS the LFSR sequence (possibly inverted) -> BM works.
2. Known whitening polynomials: descramble with each library entry and
   score the result by structure (bias + GF(2) rank evidence).
"""
from __future__ import annotations

import numpy as np

from .lfsr import LFSR, KNOWN_WHITENERS


def berlekamp_massey(bits: np.ndarray) -> tuple:
    """Returns (poly_mask, degree L). poly bit i = coefficient of x^i,
    with x^0 term always set (c0=1)."""
    s = [int(b) for b in bits]
    n = len(s)
    c = [0] * n; b = [0] * n
    c[0] = b[0] = 1
    L, m = 0, -1
    for i in range(n):
        d = s[i]
        for j in range(1, L + 1):
            d ^= c[j] & s[i - j]
        if d:
            t = c[:]
            shift = i - m
            for j in range(0, n - shift):
                c[j + shift] ^= b[j]
            if 2 * L <= i:
                L, m, b = i + 1 - L, i, t
    poly = 0
    for j in range(L + 1):
        if c[j]:
            poly |= (1 << j)
    return poly, L


def bm_to_lfsr_poly(bm_poly: int, L: int) -> int:
    """Convert the BM connection polynomial (recurrence form, s[i] =
    sum c_j s[i-j]) into the tap mask used by our Fibonacci LFSR class
    (reciprocal polynomial)."""
    out = 1 << L
    for j in range(1, L + 1):
        if (bm_poly >> j) & 1:
            out |= 1 << (L - j)
    return out


def detect_additive_scrambler(bits: np.ndarray, max_degree: int = 16,
                              window: int = 4096) -> dict:
    """Try BM on the stream head (covers preamble/idle capture).  A recovered
    LFSR is only reported when it is *short* (<= max_degree) and re-generates
    a long stretch of the observed stream."""
    head = np.asarray(bits[:window], dtype=np.uint8)
    if len(head) < 128:
        return {"found": False}
    poly, L = berlekamp_massey(head[:1024])
    if 2 <= L <= max_degree:
        lfsr_poly = bm_to_lfsr_poly(poly, L)
        seq = LFSR(lfsr_poly, seed=int(sum(int(b) << i for i, b in enumerate(head[:L]))) or 1).sequence(len(head))
        # verify by regenerating from the first L bits as state
        match = float((seq == head).mean())
        if match > 0.98:
            return {"found": True, "kind": "additive_lfsr",
                    "poly": lfsr_poly, "bm_poly": poly,
                    "degree": L, "match": match}
    return {"found": False}


def whitener_sequence(name: str, length: int) -> np.ndarray:
    """One full-period whitener sequence tiled to `length` bits."""
    w = KNOWN_WHITENERS[name]
    period = (1 << w["degree"]) - 1
    base = LFSR(w["poly"], w["seed"]).sequence(period)
    reps = length // period + 2
    return np.tile(base, reps)[:length + period], period


def _max_word_repeats(words: np.ndarray) -> int:
    """Count of the most frequent value in an integer array."""
    w = np.sort(words)
    edges = np.flatnonzero(np.diff(w)) + 1
    runs = np.diff(np.concatenate([[0], edges, [len(w)]]))
    return int(runs.max()) if len(runs) else 0


def _words(bits: np.ndarray, width: int) -> np.ndarray:
    """Integer value of every width-bit sliding window."""
    win = np.lib.stride_tricks.sliding_window_view(bits.astype(np.int64), width)
    return win @ (1 << np.arange(width - 1, -1, -1, dtype=np.int64))


def whitener_by_repeats(bits: np.ndarray, name: str, width: int = 16,
                        max_bits: int = 8192,
                        max_phases: int = None) -> dict:
    """Find a library whitener's phase without any code structure: once the
    right phase is removed, every frame's sync word reappears, so one
    width-bit word repeats about once per frame, while a still-scrambled
    stream has no word much above the random maximum (3-4 for 8k bits).
    Works for uncoded and systematic-coded (RS, LDPC) frames, where the
    GF(2) rank scan sees no period. Returns {phase, repeats, bits}."""
    bits = np.asarray(bits, dtype=np.uint8)
    n = min(len(bits), max_bits)
    seq, period = whitener_sequence(name, n)
    wr = _words(bits[:n], width)
    ws = _words(seq, width)
    best = {"phase": 0, "repeats": -1}
    for phase in range(min(period, max_phases or MAX_WHITENER_PHASES)):
        r = _max_word_repeats(wr ^ ws[phase:phase + len(wr)])
        if r > best["repeats"]:
            best = {"phase": phase, "repeats": r}
    full, _ = whitener_sequence(name, len(bits))
    best["bits"] = bits ^ full[best["phase"]:best["phase"] + len(bits)]
    return best


def raw_word_repeats(bits: np.ndarray, width: int = 16,
                     max_bits: int = 8192) -> int:
    """Baseline for whitener_by_repeats: the same statistic on the raw
    stream (high when frames are not whitened at all)."""
    return _max_word_repeats(_words(np.asarray(bits[:max_bits],
                                               dtype=np.uint8), width))


# ponytail: first 512 phases only (DVB's 2^15-1 is 64x more); sweep all
# phases or seed from a preamble if DVB-scrambled captures matter
MAX_WHITENER_PHASES = 512


def align_whitener(bits: np.ndarray, name: str, P: int,
                   rank_rows: int = 128, max_phases: int = MAX_WHITENER_PHASES,
                   stop_sig: float = None) -> dict:
    """Find the whitener phase blind.

    An additive LFSR scrambler preserves GF(2) rank structure as a
    low-degree affine overlay, so the interleaver/code period P is
    detectable *through* the scrambler; the correct descrambling phase is
    then the one that maximises the rank deficiency of the P-column
    analysis matrix (wrong phases leave a shifted-PN overlay that costs
    ~degree extra rank).
    """
    from ..gf2.rank import gf2_rank_bits, expected_random_deficiency
    bits = np.asarray(bits, dtype=np.uint8)
    seq, period = whitener_sequence(name, len(bits))
    n_rows = min(len(bits) // P, rank_rows)
    best = {"phase": 0, "sig": -1.0}
    # long-period whiteners (e.g. DVB, 2^15-1) cannot be exhaustively
    # phase-searched at interactive speed; the pipeline warns when the
    # chain ends unscrambled and unvalidated
    n_phases = min(period, max_phases)
    for phase in range(n_phases):
        x = bits ^ seq[phase:phase + len(bits)]
        r, used = gf2_rank_bits(x, P, n_rows)
        sig = (min(used, P) - r) - expected_random_deficiency(used, P)
        if sig > best["sig"]:
            best = {"phase": phase, "sig": float(sig)}
            if stop_sig is not None and sig >= stop_sig:
                break
    best["bits"] = bits ^ seq[best["phase"]:best["phase"] + len(bits)]
    best["phases_searched"] = n_phases
    best["period"] = period
    return best
