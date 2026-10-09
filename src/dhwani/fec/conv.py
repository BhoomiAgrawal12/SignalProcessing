"""Convolutional codes: encoder + vectorised Viterbi decoder.

Generator polynomials are given in octal (e.g. (0o171, 0o133) for the
CCSDS/Voyager rate-1/2 K=7 code).  Convention: MSB of the polynomial taps
the *current* input bit; shift register holds the K-1 previous bits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConvCode:
    K: int                 # constraint length
    generators: tuple      # octal ints, one per output stream (rate 1/n)

    @property
    def n_out(self) -> int:
        return len(self.generators)

    @property
    def rate(self) -> float:
        return 1.0 / self.n_out


# The codes an analyst actually meets in the wild (gr-satellites / CCSDS).
STANDARD_CODES = [
    ConvCode(7, (0o171, 0o133)),   # CCSDS / Voyager / DVB
    ConvCode(7, (0o133, 0o171)),   # swapped order
    ConvCode(9, (0o561, 0o753)),   # IS-95
    ConvCode(9, (0o753, 0o561)),
    ConvCode(5, (0o35, 0o23)),
    ConvCode(3, (0o7, 0o5)),
]


def _output_table(code: ConvCode) -> np.ndarray:
    """out[state, inbit] -> packed output bits (int), state = previous K-1 bits,
    MSB = most recent."""
    K, gens = code.K, code.generators
    n_states = 1 << (K - 1)
    table = np.zeros((n_states, 2), dtype=np.uint8)
    for s in range(n_states):
        for b in (0, 1):
            reg = (b << (K - 1)) | s          # current bit on top of history
            out = 0
            for g in gens:
                out = (out << 1) | (bin(reg & g).count("1") & 1)
            table[s, b] = out
    return table


def _next_state(state: int, bit: int, K: int) -> int:
    return ((bit << (K - 1)) | state) >> 1


def conv_encode(bits: np.ndarray, code: ConvCode, terminate: bool = True) -> np.ndarray:
    """Encode; returns interleaved output streams (v0[0], v1[0], v0[1], ...)."""
    table = _output_table(code)
    K = code.K
    n = code.n_out
    tail = (K - 1) if terminate else 0
    out = np.empty((len(bits) + tail) * n, dtype=np.uint8)
    state = 0
    src = np.concatenate([np.asarray(bits, dtype=np.uint8),
                          np.zeros(tail, dtype=np.uint8)])
    for i, b in enumerate(src):
        o = table[state, b]
        for j in range(n):
            out[i * n + j] = (o >> (n - 1 - j)) & 1
        state = _next_state(state, int(b), K)
    return out


def viterbi_decode(bits: np.ndarray, code: ConvCode, soft: np.ndarray = None) -> np.ndarray:
    """Hard/soft-decision Viterbi, vectorised add-compare-select.

    `bits`: interleaved coded bits.  `soft` (optional): LLRs aligned with
    bits (positive => bit 0 more likely); when given, branch metrics use
    them, which markedly improves low-SNR decoding.
    Returns decoded bits (including any tail bits; caller trims).
    """
    K, n = code.K, code.n_out
    n_states = 1 << (K - 1)
    n_steps = len(bits) // n
    if n_steps == 0:
        return np.zeros(0, dtype=np.uint8)
    table = _output_table(code)

    # branch output bits per (state, input): (S, 2, n)
    outbits = np.zeros((n_states, 2, n), dtype=np.int8)
    for j in range(n):
        outbits[:, :, j] = (table >> (n - 1 - j)) & 1

    # Predecessor structure: for next state ns, the input bit that led there
    # is ns >> (K-2); the two possible previous states are 2*(ns & mask) and
    # 2*(ns & mask) + 1.
    ns_arr = np.arange(n_states)
    in_bit = (ns_arr >> (K - 2)).astype(np.intp) if K > 2 else ns_arr.astype(np.intp)
    mask = (1 << (K - 2)) - 1 if K > 2 else 0
    prev0 = ((ns_arr & mask) << 1).astype(np.intp)
    prev1 = prev0 + 1

    INF = 1e18
    pm = np.full(n_states, INF)
    pm[0] = 0.0
    bp = np.zeros((n_steps, n_states), dtype=np.uint8)  # 0 -> prev0, 1 -> prev1

    rx = np.asarray(bits[: n_steps * n], dtype=np.int8).reshape(n_steps, n)
    soft_arr = None
    if soft is not None:
        soft_arr = np.asarray(soft[: n_steps * n], dtype=np.float64).reshape(n_steps, n)

    out0 = outbits[prev0, in_bit]     # (S, n) output bits along branch prev0->ns
    out1 = outbits[prev1, in_bit]
    for t in range(n_steps):
        if soft_arr is None:
            bm0 = (out0 != rx[t]).sum(axis=1)
            bm1 = (out1 != rx[t]).sum(axis=1)
        else:
            llr = soft_arr[t]
            bm0 = np.where(out0 == 1, llr, 0.0).sum(axis=1) - np.where(out0 == 0, llr, 0.0).sum(axis=1)
            bm1 = np.where(out1 == 1, llr, 0.0).sum(axis=1) - np.where(out1 == 0, llr, 0.0).sum(axis=1)
        c0 = pm[prev0] + bm0
        c1 = pm[prev1] + bm1
        choose1 = c1 < c0
        pm = np.where(choose1, c1, c0)
        bp[t] = choose1
    state = int(np.argmin(pm))
    decoded = np.zeros(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = state >> (K - 2) if K > 2 else state
        p_ = prev1[state] if bp[t, state] else prev0[state]
        state = int(p_)
    return decoded
