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

from .lfsr import LFSR, additive_scramble, KNOWN_WHITENERS


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


def _structure_score(bits: np.ndarray) -> float:
    """Cheap structure metric: deviation of run statistics + bias from
    ideal randomness. Higher = more structured (i.e. descrambling helped)."""
    bits = np.asarray(bits, dtype=np.int8)
    n = len(bits)
    if n < 64:
        return 0.0
    bias = abs(bits.mean() - 0.5)
    # lag-correlation structure at small lags
    x = bits * 2 - 1
    score = 0.0
    for lag in (1, 2, 4, 8, 16, 32):
        if n > lag:
            score = max(score, abs(np.mean(x[:-lag] * x[lag:])))
    return float(bias * 2 + score)


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


def try_known_whiteners(bits: np.ndarray) -> list:
    """Descramble with each library whitener; return options ranked by how
    much structure the descrambling reveals. The 'none' option is always
    included so downstream stages can pick the raw stream."""
    base = _structure_score(bits)
    results = [{"name": "none", "bits": np.asarray(bits, dtype=np.uint8),
                "score": base, "poly": None, "seed": None}]
    for name, w in KNOWN_WHITENERS.items():
        out = additive_scramble(bits, w["poly"], w["seed"])
        results.append({"name": name, "bits": out,
                        "score": _structure_score(out),
                        "poly": w["poly"], "seed": w["seed"]})
    results.sort(key=lambda r: -r["score"])
    return results


def whitener_sequence(name: str, length: int) -> np.ndarray:
    """One full-period whitener sequence tiled to `length` bits."""
    w = KNOWN_WHITENERS[name]
    period = (1 << w["degree"]) - 1
    base = LFSR(w["poly"], w["seed"]).sequence(period)
    reps = length // period + 2
    return np.tile(base, reps)[:length + period], period


def align_whitener(bits: np.ndarray, name: str, P: int,
                   rank_rows: int = 128, max_phases: int = 512,
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
    # phase-searched at interactive speed; the truncated search is
    # reported honestly via "phases_searched"
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
