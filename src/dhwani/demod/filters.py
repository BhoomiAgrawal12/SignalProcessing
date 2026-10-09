"""Pulse-shaping filters shared by modulator and demodulator."""
from __future__ import annotations

import numpy as np


def rrc_taps(sps: float, span: int, beta: float) -> np.ndarray:
    """Root-raised-cosine filter taps (unit energy)."""
    n = int(span * sps)
    if n % 2 == 0:
        n += 1
    t = (np.arange(n) - n // 2) / sps
    taps = np.zeros(n)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            taps[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(4 * beta * ti) - 1.0) < 1e-9:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            taps[i] = (np.sin(np.pi * ti * (1 - beta)) +
                       4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / \
                      (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return taps / np.sqrt((taps ** 2).sum())
