"""LFSR sequence generation and additive (synchronous) scrambling.

Polynomial convention: integer bitmask over x^0..x^deg, LSB = x^0 term.
E.g. PN9 x^9+x^5+1 -> 0b1000100001 = 0x221.
Fibonacci form: new bit = XOR of tapped register bits; register shifts right.
"""
from __future__ import annotations

import numpy as np


class LFSR:
    def __init__(self, poly: int, seed: int, degree: int = None):
        self.poly = poly
        self.degree = degree or (poly.bit_length() - 1)
        if seed == 0:
            seed = (1 << self.degree) - 1
        self.state = seed & ((1 << self.degree) - 1)
        # tap positions: bits of poly excluding x^degree term
        self.taps = [i for i in range(self.degree)
                     if (poly >> i) & 1] or [0]

    def sequence(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.uint8)
        state = self.state
        deg = self.degree
        taps = self.taps
        for i in range(n):
            out[i] = state & 1
            fb = 0
            for t in taps:
                fb ^= (state >> t) & 1
            state = (state >> 1) | (fb << (deg - 1))
        self.state = state
        return out


def additive_scramble(bits: np.ndarray, poly: int, seed: int) -> np.ndarray:
    """XOR the stream with the LFSR output (self-inverse)."""
    seq = LFSR(poly, seed).sequence(len(bits))
    return (np.asarray(bits, dtype=np.uint8) ^ seq).astype(np.uint8)


# Common whitening sequences seen in real waveforms (report S7c).
KNOWN_WHITENERS = {
    "PN9-CC1101": {"poly": 0x221, "seed": 0x1FF, "degree": 9},
    "CCSDS":      {"poly": 0x1A9, "seed": 0xFF,  "degree": 8},   # x^8+x^7+x^5+x^3+1
    "DVB":        {"poly": 0xA011, "seed": 0x4A80, "degree": 15}, # x^15+x^14+1 approx conventions vary
    "IEEE802.11": {"poly": 0x91,  "seed": 0x7F,  "degree": 7},   # x^7+x^4+1
}
