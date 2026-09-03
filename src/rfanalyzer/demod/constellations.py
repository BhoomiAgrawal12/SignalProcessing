"""Constellation definitions with Gray bit mappings, shared by the
modulator (synth factory) and the demodulator/slicer."""
from __future__ import annotations

import numpy as np


def _psk(order: int) -> tuple:
    """Gray-coded PSK. Returns (points, bits_per_symbol, labels)."""
    k = int(np.log2(order))
    gray = np.arange(order) ^ (np.arange(order) >> 1)
    pts = np.exp(1j * (2 * np.pi * np.arange(order) / order +
                       (np.pi / 4 if order == 4 else 0)))
    labels = np.zeros((order, k), dtype=np.uint8)
    points = np.zeros(order, dtype=np.complex128)
    for idx in range(order):
        g = gray[idx]
        points[g] = pts[idx] if False else pts[idx]
        # store mapping label->point: label g at angle position idx
    # build label -> point table directly
    table = np.zeros(order, dtype=np.complex128)
    for pos in range(order):
        table[gray[pos]] = pts[pos]
    for lab in range(order):
        labels[lab] = [(lab >> (k - 1 - j)) & 1 for j in range(k)]
    return table, k


def _qam(order: int) -> tuple:
    side = int(np.sqrt(order))
    k = int(np.log2(order))
    kb = k // 2
    gray = np.arange(side) ^ (np.arange(side) >> 1)
    # gray[i] is the gray code of index i; we need position of each label
    pos_of = np.argsort(gray)
    levels = 2 * np.arange(side) - (side - 1)
    table = np.zeros(order, dtype=np.complex128)
    for lab in range(order):
        li = (lab >> kb) & ((1 << kb) - 1)
        lq = lab & ((1 << kb) - 1)
        table[lab] = levels[pos_of[li]] + 1j * levels[pos_of[lq]]
    table /= np.sqrt((np.abs(table) ** 2).mean())
    return table, k


CONSTELLATIONS = {
    "BPSK": _psk(2),
    "QPSK": _psk(4),
    "8PSK": _psk(8),
    "16QAM": _qam(16),
    "64QAM": _qam(64),
}


def bits_to_iq_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    table, k = CONSTELLATIONS[modulation]
    n_sym = len(bits) // k
    b = np.asarray(bits[: n_sym * k], dtype=np.int64).reshape(n_sym, k)
    weights = 1 << np.arange(k - 1, -1, -1)
    labels = (b * weights).sum(axis=1)
    return table[labels]


def slice_symbols(symbols: np.ndarray, modulation: str,
                  noise_var: float = 0.1) -> tuple:
    """Nearest-point slicing + max-log LLRs.

    Returns (hard_bits, llrs, evm_percent). LLR > 0 means bit 0 more
    likely (convention used throughout the bit layer)."""
    table, k = CONSTELLATIONS[modulation]
    d2 = np.abs(symbols[:, None] - table[None, :]) ** 2   # (N, order)
    nearest = np.argmin(d2, axis=1)
    err = symbols - table[nearest]
    evm = float(np.sqrt((np.abs(err) ** 2).mean() /
                        (np.abs(table) ** 2).mean()) * 100)
    order = len(table)
    labels = np.arange(order)
    hard = np.zeros((len(symbols), k), dtype=np.uint8)
    llrs = np.zeros((len(symbols), k), dtype=np.float32)
    nv = max(noise_var, 1e-6)
    for j in range(k):
        bit_of_label = (labels >> (k - 1 - j)) & 1
        d0 = d2[:, bit_of_label == 0].min(axis=1)
        d1 = d2[:, bit_of_label == 1].min(axis=1)
        llrs[:, j] = (d1 - d0) / nv
        hard[:, j] = (llrs[:, j] < 0)
    return hard.ravel(), llrs.ravel(), evm
