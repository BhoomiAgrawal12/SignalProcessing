"""Bit packing utilities: 64 bits per uint64 word.

All GF(2) heavy lifting in this project runs on packed words - a ~64x
speed-up over per-bit arithmetic.
Bit i of the stream maps to bit (i % 64) of word (i // 64), LSB-first.
"""
from __future__ import annotations

import numpy as np

_POP = np.array([bin(i).count("1") for i in range(65536)], dtype=np.uint8)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack a 0/1 uint8 array into uint64 words (LSB-first)."""
    bits = np.ascontiguousarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 64
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little").view(np.uint64)


def pack_rows(bits: np.ndarray, L: int) -> np.ndarray:
    """Reshape a bit stream into rows of length L and pack each row.

    Returns array of shape (n_rows, ceil(L/64)) uint64."""
    n_rows = len(bits) // L
    rows = np.ascontiguousarray(bits[: n_rows * L], dtype=np.uint8).reshape(n_rows, L)
    padded_L = ((L + 63) // 64) * 64
    if padded_L != L:
        rows = np.concatenate(
            [rows, np.zeros((n_rows, padded_L - L), dtype=np.uint8)], axis=1)
    packed = np.packbits(rows, axis=1, bitorder="little")
    return packed.view(np.uint64).reshape(n_rows, padded_L // 64)


def unpack_bits(words: np.ndarray, n_bits: int) -> np.ndarray:
    """Inverse of pack_bits."""
    b = np.ascontiguousarray(words).view(np.uint8)
    bits = np.unpackbits(b, bitorder="little")
    return bits[:n_bits].astype(np.uint8)


def popcount_u64(words: np.ndarray) -> np.ndarray:
    """Per-word popcount via 16-bit lookup table."""
    v = np.ascontiguousarray(words).view(np.uint16).reshape(*words.shape, 4)
    return _POP[v].sum(axis=-1).astype(np.int64)
