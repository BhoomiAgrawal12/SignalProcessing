"""GF(2) linear algebra on bit-packed uint64 matrices.

The rank-deficiency scan is the detector that powers both blind
de-interleaving (S8) and blind FEC identification (S9): rows of a matrix
whose width equals a multiple of the code/interleaver period are confined
to a k-dimensional subspace, so the matrix rank drops below full - while a
wrong width gives essentially full rank (Sicot & Houcke 2009).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..bits.packing import pack_rows


def gf2_rank_packed(M: np.ndarray, n_cols: int) -> int:
    """Rank over GF(2) of a bit-packed matrix (rows x words), in-place-free.

    Gaussian elimination with XOR row updates, vectorised across rows.
    """
    M = M.copy()
    n_rows = M.shape[0]
    rank = 0
    for col in range(n_cols):
        if rank >= n_rows:
            break
        w, b = col >> 6, np.uint64(1 << (col & 63))
        # find pivot at or below `rank`
        col_bits = (M[rank:, w] & b) != 0
        nz = np.nonzero(col_bits)[0]
        if nz.size == 0:
            continue
        piv = rank + nz[0]
        if piv != rank:
            M[[rank, piv]] = M[[piv, rank]]
        # eliminate this column from every other row that has it set
        mask = (M[:, w] & b) != 0
        mask[rank] = False
        if mask.any():
            M[mask] ^= M[rank]
        rank += 1
    return rank


def gf2_rank_bits(bits: np.ndarray, L: int, n_rows: Optional[int] = None,
                  offset: int = 0) -> tuple:
    """Rank of the matrix formed by reshaping `bits[offset:]` into rows of
    length L. Returns (rank, n_rows_used)."""
    usable = bits[offset:]
    max_rows = len(usable) // L
    if n_rows is None:
        n_rows = max_rows
    n_rows = min(n_rows, max_rows)
    if n_rows < 2:
        return 0, 0
    M = pack_rows(usable[: n_rows * L], L)
    return gf2_rank_packed(M, L), n_rows


def expected_random_deficiency(n_rows: int, n_cols: int) -> float:
    """Expected rank deficiency of a uniformly random GF(2) matrix.

    For an n x m random binary matrix with n >= m the expected deficiency is
    sum_{j>=1} 2^{-j(n-m+j)} approximately; for square-ish matrices it is
    ~0.85 at n==m and decays geometrically with |n-m|.  We use the standard
    series truncated at 10 terms."""
    d = abs(n_rows - n_cols)
    e = 0.0
    for j in range(1, 11):
        exponent = j * (d + j)
        if exponent > 60:
            break
        e += 1.0 / (2.0 ** exponent)
    return e


def rank_profile(bits: np.ndarray, L_min: int = 2, L_max: int = 512,
                 rows_factor: int = 2, extra_rows: int = 32,
                 max_bits: int = 200000, L_values: Optional[np.ndarray] = None,
                 progress=None) -> dict:
    """Compute normalised rank deficiency for every trial row length L.

    Returns dict with arrays: L, rank, rows, deficiency, significance -
    where significance = (observed deficiency - expected random deficiency).
    A large significance at L is evidence of linear structure with period
    dividing L.
    """
    if len(bits) > max_bits:
        bits = bits[:max_bits]
    if L_values is None:
        L_values = np.arange(L_min, L_max + 1)
    out_L, out_rank, out_rows, out_def, out_sig = [], [], [], [], []
    for i, L in enumerate(L_values):
        L = int(L)
        n_rows = min(len(bits) // L, rows_factor * L + extra_rows)
        if n_rows < max(8, L // 4):
            # not enough data for a meaningful rank at this width
            continue
        r, used = gf2_rank_bits(bits, L, n_rows)
        full = min(used, L)
        deficiency = full - r
        sig = deficiency - expected_random_deficiency(used, L)
        out_L.append(L); out_rank.append(r); out_rows.append(used)
        out_def.append(deficiency); out_sig.append(sig)
        if progress is not None and i % 16 == 0:
            progress(i / len(L_values))
    return {"L": np.array(out_L), "rank": np.array(out_rank),
            "rows": np.array(out_rows), "deficiency": np.array(out_def),
            "significance": np.array(out_sig, dtype=float)}
