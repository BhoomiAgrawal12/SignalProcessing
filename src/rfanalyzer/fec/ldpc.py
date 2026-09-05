"""LDPC support: candidate-set identification plus bit-flipping decode.

Blind reconstruction of an arbitrary sparse parity-check matrix from a
noisy stream is research-grade (report S9: "identify LDPC codes from a
candidate set of known standards rather than reconstructing an arbitrary
H from scratch - say this openly"). This module therefore implements the
professional candidate-set approach:

* LDPCCode builds a systematic code H = [P | I_m] with a sparse,
  deterministic P (seeded), so the encoder, the factory, and the
  identifier all share exactly the same matrix.
* identify_ldpc sweeps the configured candidates over the codeword
  alignment and scores each by the all-zero-syndrome fraction, the same
  objective confidence metric used for the other code families.
* decode is a bounded bit-flipping decoder, adequate for the moderate
  error rates where rank-based blind analysis works at all.

Standardised matrices (CCSDS, DVB-S2, 802.11n) can be added to the
candidate list by loading their H into LDPCCode.from_parity_check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class LDPCCode:
    """Systematic LDPC-style linear code: H = [P | I_m], G = [I_k | P^T]."""

    def __init__(self, n: int, k: int, seed: int = 1, column_weight: int = 3):
        if not 0 < k < n:
            raise ValueError("need 0 < k < n")
        self.n, self.k, self.seed = n, k, seed
        self.m = n - k
        rng = np.random.default_rng(seed)
        P = np.zeros((self.m, k), dtype=np.uint8)
        for col in range(k):
            rows = rng.choice(self.m, size=min(column_weight, self.m),
                              replace=False)
            P[rows, col] = 1
        # every check row must involve at least one information bit
        for row in range(self.m):
            if not P[row].any():
                P[row, rng.integers(0, k)] = 1
        self.P = P
        self.H = np.concatenate([P, np.eye(self.m, dtype=np.uint8)], axis=1)

    @classmethod
    def from_parity_check(cls, H: np.ndarray, name: str = "custom"):
        obj = cls.__new__(cls)
        H = np.asarray(H, dtype=np.uint8)
        obj.H = H
        obj.m, obj.n = H.shape
        obj.k = obj.n - obj.m
        obj.P = None
        obj.seed = name
        return obj

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode blocks of k bits; pads the tail block with zeros."""
        bits = np.asarray(info_bits, dtype=np.uint8)
        pad = (-len(bits)) % self.k
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        blocks = bits.reshape(-1, self.k)
        parity = (blocks @ self.P.T) % 2
        return np.concatenate([blocks, parity], axis=1).reshape(-1).astype(np.uint8)

    def syndromes(self, codewords: np.ndarray) -> np.ndarray:
        """(n_cw, m) syndrome matrix for stacked codewords."""
        return (np.asarray(codewords, dtype=np.uint8) @ self.H.T) % 2

    def syndrome_zero_rate(self, bits: np.ndarray, offset: int = 0,
                           max_codewords: int = 64) -> float:
        b = np.asarray(bits[offset:], dtype=np.uint8)
        n_cw = min(len(b) // self.n, max_codewords)
        if n_cw < 2:
            return 0.0
        cw = b[: n_cw * self.n].reshape(n_cw, self.n)
        return float((self.syndromes(cw) == 0).all(axis=1).mean())

    def decode(self, codeword: np.ndarray, max_iters: int = 30) -> tuple:
        """Bit-flipping decode of one n-bit block.

        Returns (info_bits, converged)."""
        c = np.asarray(codeword, dtype=np.uint8).copy()
        for _ in range(max_iters):
            syn = (self.H @ c) % 2
            if not syn.any():
                return c[: self.k], True
            # count of unsatisfied checks per bit
            votes = self.H.T @ syn
            worst = votes.max()
            if worst == 0:
                break
            c[votes == worst] ^= 1
        syn = (self.H @ c) % 2
        return c[: self.k], not syn.any()

    def decode_stream(self, bits: np.ndarray, offset: int = 0,
                      max_blocks: int = 4096) -> tuple:
        """Decode consecutive blocks; returns (info_bits, converged_rate)."""
        b = np.asarray(bits[offset:], dtype=np.uint8)
        n_blocks = min(len(b) // self.n, max_blocks)
        out, ok = [], 0
        for i in range(n_blocks):
            info, converged = self.decode(b[i * self.n:(i + 1) * self.n])
            out.append(info)
            ok += converged
        if not out:
            return np.zeros(0, dtype=np.uint8), 0.0
        return np.concatenate(out), ok / n_blocks


def identify_ldpc(bits: np.ndarray, candidates: list,
                  quick_codewords: int = 8, min_rate: float = 0.5) -> list:
    """Sweep candidate LDPC codes over the codeword alignment.

    candidates: [{"n": int, "k": int, "seed": int}, ...]
    Returns hits sorted by syndrome-zero rate."""
    bits = np.asarray(bits, dtype=np.uint8)
    hits = []
    for cand in candidates:
        code = LDPCCode(cand["n"], cand["k"], cand.get("seed", 1))
        if len(bits) < 2 * code.n:
            continue
        best = None
        max_off = min(code.n, max(1, len(bits) - 2 * code.n) + 1)
        for off in range(max_off):
            rate = code.syndrome_zero_rate(bits, off, quick_codewords)
            if rate > min_rate and (best is None or rate > best["rate"]):
                best = {"offset": off, "rate": rate}
                if rate == 1.0:
                    break
        if best:
            confirm = code.syndrome_zero_rate(bits, best["offset"], 64)
            hits.append({**cand, "offset": best["offset"],
                         "syndrome_zero_rate": confirm, "code": code})
    hits.sort(key=lambda h: -h["syndrome_zero_rate"])
    return hits
