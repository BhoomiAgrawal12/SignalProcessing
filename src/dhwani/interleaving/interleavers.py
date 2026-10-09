"""Interleaver encoders + exact inverses (used by the synthetic factory and
by the blind identifier to test hypotheses)."""
from __future__ import annotations

import numpy as np


def block_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Write row-wise, read column-wise, block by block."""
    P = rows * cols
    n = (len(bits) // P) * P
    m = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, rows, cols)
    return m.transpose(0, 2, 1).reshape(-1)


def block_deinterleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    P = rows * cols
    n = (len(bits) // P) * P
    m = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, cols, rows)
    return m.transpose(0, 2, 1).reshape(-1)


def _conv_perm(n: int, branches: int, delay: int) -> np.ndarray:
    """Output position for each input position of a multiplexed convolutional
    interleaver (branch b delays by b*delay symbols); realised as a fixed
    permutation over a stream of length n (positions that would come from
    'before the start' wrap around, which keeps it invertible for testing)."""
    idx = np.arange(n)
    b = idx % branches
    out = idx + b * branches * delay
    return np.argsort(out % n, kind="stable")


def conv_interleave(bits: np.ndarray, branches: int, delay: int) -> np.ndarray:
    n = len(bits)
    perm = _conv_perm(n, branches, delay)
    return np.asarray(bits, dtype=np.uint8)[perm]


def conv_deinterleave(bits: np.ndarray, branches: int, delay: int) -> np.ndarray:
    n = len(bits)
    perm = _conv_perm(n, branches, delay)
    out = np.empty(n, dtype=np.uint8)
    out[perm] = np.asarray(bits, dtype=np.uint8)
    return out


def helical_interleave(bits: np.ndarray, rows: int, cols: int, step: int) -> np.ndarray:
    """Block interleaver composed with a per-column cyclic shift of `step`
    (diagonal/helical read-out)."""
    P = rows * cols
    n = (len(bits) // P) * P
    m = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, rows, cols)
    for c in range(cols):
        m[:, :, c] = np.roll(m[:, :, c], (c * step) % rows, axis=1)
    return m.transpose(0, 2, 1).reshape(-1)


def helical_deinterleave(bits: np.ndarray, rows: int, cols: int, step: int) -> np.ndarray:
    P = rows * cols
    n = (len(bits) // P) * P
    m = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, cols, rows).transpose(0, 2, 1).copy()
    for c in range(cols):
        m[:, :, c] = np.roll(m[:, :, c], -((c * step) % rows), axis=1)
    return m.reshape(len(m), -1).reshape(-1)


def pn_interleave(bits: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    P = len(permutation)
    n = (len(bits) // P) * P
    m = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, P)
    return m[:, permutation].reshape(-1)


def pn_deinterleave(bits: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    P = len(permutation)
    inv = np.argsort(permutation)
    n = (len(bits) // P) * P
    m = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, P)
    return m[:, inv].reshape(-1)
