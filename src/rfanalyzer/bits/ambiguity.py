"""Stage S7a: phase/IQ/polarity ambiguity fan-out.

A blind carrier loop locks to *a* constellation orientation.  We generate
every candidate interpretation and let the objective bit-layer tests pick
the winner (this is why the pipeline is a search, not a line)."""
from __future__ import annotations

import numpy as np

from ..common.models import BitStream
from ..demod.constellations import CONSTELLATIONS, slice_symbols


def _differential_decode(bits: np.ndarray) -> np.ndarray:
    d = np.bitwise_xor(bits[1:], bits[:-1])
    return np.concatenate([[bits[0]], d]).astype(np.uint8)


def enumerate_ambiguities(symbols: np.ndarray, modulation: str,
                          max_streams: int = 32,
                          include_differential: bool = True) -> list:
    """Returns list[BitStream] covering rotations x conjugation
    (x differential decode)."""
    if modulation not in CONSTELLATIONS or symbols is None or len(symbols) == 0:
        return []
    order = {"BPSK": 2, "QPSK": 4, "8PSK": 8}.get(modulation, 4)
    n_rot = order if modulation in ("BPSK", "QPSK", "8PSK") else 4
    streams = []
    noise_var = max(1e-4, float(np.var(np.abs(symbols)) * 0.5))
    for conj in (False, True):
        base = np.conj(symbols) if conj else symbols
        for r in range(n_rot):
            rot = base * np.exp(2j * np.pi * r / n_rot)
            hard, llrs, _ = slice_symbols(rot, modulation, noise_var)
            hyp = {"rotation": f"{r}/{n_rot} turn", "iq_swap": conj,
                   "differential": False}
            streams.append(BitStream(bits=hard, llrs=llrs, hypothesis=hyp))
            if include_differential:
                streams.append(BitStream(
                    bits=_differential_decode(hard),
                    hypothesis={**hyp, "differential": True}))
            if len(streams) >= max_streams:
                return streams
    return streams
