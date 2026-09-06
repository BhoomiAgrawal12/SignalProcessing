"""Engine A: higher-order-cumulant modulation classifier (explainable).

Reference feature vectors are computed *exactly* from the constellation
tables with the same estimator formulas used on the data, so estimator
bias cancels by construction.  Features are rotation-invariant moduli;
carrier offset must be removed first (the driver tries the plausible
M-power corrections and keeps the most structured result).
"""
from __future__ import annotations

import numpy as np

from ..demod.constellations import CONSTELLATIONS


def _features(s: np.ndarray) -> np.ndarray:
    """[|C20|, |C40|, |C42|, m63] on unit-power samples."""
    s = s / (np.sqrt((np.abs(s) ** 2).mean()) + 1e-12)
    m20 = (s ** 2).mean()
    m21 = (np.abs(s) ** 2).mean()
    m40 = (s ** 4).mean()
    m42 = (np.abs(s) ** 4).mean()
    m63 = (np.abs(s) ** 6).mean()
    c20 = m20 / m21
    c40 = (m40 - 3 * m20 ** 2) / m21 ** 2
    c42 = (m42 - np.abs(m20) ** 2 - 2 * m21 ** 2) / m21 ** 2
    return np.array([np.abs(c20), np.abs(c40), np.abs(c42), m63 / m21 ** 3])


_REFERENCE = {}
for _name, (_table, _k) in CONSTELLATIONS.items():
    _REFERENCE[_name] = _features(_table.astype(np.complex128))

# feature weights: |C20| separates BPSK sharply; m63 separates QAM orders
_WEIGHTS = np.array([3.0, 2.0, 2.0, 1.0])


def classify_cumulants(symbols: np.ndarray, snr_db: float = None) -> dict:
    """Classify symbol-spaced, CFO-corrected samples.

    Returns {"probabilities": {label: p}, "features": [...],
             "reference": {...}} ranked by probability."""
    f = _features(np.asarray(symbols, dtype=np.complex128))
    d2 = {}
    for name, ref in _REFERENCE.items():
        d2[name] = float((((f - ref) * _WEIGHTS) ** 2).sum())
    # absolute-fit guard: softmax only sees RELATIVE distances, so a
    # Gaussian cloud (OFDM, noise) would otherwise claim its least-bad
    # reference with false certainty; a poor best fit deflates all
    # probabilities and the mass moves to an explicit no-fit bucket
    d_best = min(d2.values())
    fit_penalty = float(np.exp(-max(0.0, d_best - 2.0) / 1.0))
    # temperature scaled by noise level: at low SNR features blur, so
    # soften the decision rather than overclaim
    temp = 0.05
    if snr_db is not None and snr_db < 15:
        temp = 0.05 * (1 + (15 - snr_db) / 5)
    logits = {k: -v / temp for k, v in d2.items()}
    mx = max(logits.values())
    exps = {k: np.exp(v - mx) for k, v in logits.items()}
    z = sum(exps.values())
    probs = {k: float(v / z * fit_penalty) for k, v in
             sorted(exps.items(), key=lambda kv: -kv[1])}
    return {"probabilities": probs,
            "fit_distance": round(d_best, 3),
            "no_constellation_mass": round(1.0 - fit_penalty, 3),
            "features": [round(float(x), 4) for x in f],
            "feature_names": ["|C20|", "|C40|", "|C42|", "m63"],
            "reference": {k: [round(float(x), 3) for x in v]
                          for k, v in _REFERENCE.items()}}
