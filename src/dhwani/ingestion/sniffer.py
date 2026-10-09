"""Automatic format sniffing for headerless raw .iq files (stage S0).

Tests, in order: file-size divisibility, per-dtype value histograms,
endianness plausibility, real-vs-complex spectral symmetry.  Produces a
ranked list of candidate interpretations with a confidence for each and a
human-readable explanation.  NEVER invents a sample rate.
"""
from __future__ import annotations

import os

import numpy as np

CANDIDATE_DTYPES = [
    ("complex64", np.complex64, 8),
    ("float32", np.float32, 4),
    ("int16", np.int16, 2),
    ("int8", np.int8, 1),
    ("uint8", np.uint8, 1),
]


def _load_head(path: str, dtype, endian: str, n_values: int = 262144) -> np.ndarray:
    dt = np.dtype(dtype).newbyteorder("<" if endian == "little" else ">")
    count = min(n_values, os.path.getsize(path) // dt.itemsize)
    return np.fromfile(path, dtype=dt, count=count)


def _histogram_score(vals: np.ndarray, name: str) -> float:
    """How plausible do these numbers look as radio samples?"""
    if len(vals) == 0:
        return 0.0
    finite = np.isfinite(vals)
    if finite.mean() < 0.999:
        return 0.0
    v = vals[finite].astype(np.float64)
    if v.std() == 0:
        return 0.05
    score = 1.0
    if name == "uint8":
        centre = v.mean()
        score *= np.exp(-abs(centre - 127.5) / 40)
    elif name == "int8":
        score *= np.exp(-abs(v.mean()) / 20)
    elif name == "int16":
        score *= np.exp(-abs(v.mean()) / 3000)
        # int16 data usually doesn't hug full scale continuously
        score *= 1.0 - 0.5 * float((np.abs(v) > 32000).mean())
    elif name in ("float32", "complex64"):
        m = np.abs(v).max()
        if m == 0 or m > 1e6 or np.abs(v).mean() < 1e-30:
            return 0.02
        score *= 1.0 if m <= 100 else 0.3
    # crest factor sanity: radio noise+signal is not constant amplitude
    crest = np.abs(v).max() / (np.abs(v).std() + 1e-12)
    if crest < 1.5 or crest > 1e4:
        score *= 0.3
    return float(score)


def _spectral_symmetry(x: np.ndarray) -> float:
    """1.0 = perfectly symmetric PSD about 0 (may indicate real-valued
    data), 0 = asymmetric. Reported as an annotation only - genuine IQ
    signals (e.g. FSK) can be symmetric too, so this must not penalise."""
    n = min(len(x), 65536)
    if n < 1024:
        return 0.0
    X = np.fft.fftshift(np.abs(np.fft.fft(x[:n] * np.hanning(n))) ** 2)
    half = n // 2
    left = X[1:half][::-1]
    right = X[half + 1: 2 * half]
    m = min(len(left), len(right))
    num = np.minimum(left[:m], right[:m]).sum()
    den = np.maximum(left[:m], right[:m]).sum() + 1e-12
    return float(num / den)


def _spectral_structure(x: np.ndarray) -> float:
    """0..1: how much band structure the spectrum shows. The correct
    sample-format interpretation of a real recording is band-limited or
    peaky; a wrong dtype/endianness reads as white noise (flat PSD)."""
    n = min(len(x), 65536)
    if n < 1024:
        return 0.0
    X = np.abs(np.fft.fft(x[:n] * np.hanning(n))) ** 2
    psd_db = 10 * np.log10(X + 1e-20)
    # max-median catches even a single narrow tone; white noise gives a
    # crest of roughly 10-13 dB from extreme-value statistics alone, so
    # that floor is subtracted before scaling
    crest = float(psd_db.max() - np.median(psd_db))
    return float(np.clip((crest - 13.0) / 25.0, 0.0, 1.0))


def sniff_raw_iq(path: str) -> list:
    """Return ranked candidate interpretations:
    [{dtype, endian, complex, confidence, explanation}, ...]"""
    size = os.path.getsize(path)
    results = []
    for name, dtype, itemsize in CANDIDATE_DTYPES:
        pair = 2 * itemsize if name != "complex64" else itemsize
        if size % pair:
            continue    # file-size test: must hold an integer number of IQ pairs
        endians = ["little"] if itemsize == 1 or name == "complex64" else ["little", "big"]
        for endian in endians:
            try:
                vals = _load_head(path, dtype, endian)
            except Exception:
                continue
            with np.errstate(invalid="ignore", over="ignore"):
                if name == "complex64":
                    flat = np.concatenate([vals.real, vals.imag])
                else:
                    flat = vals.astype(np.float64)
            hscore = _histogram_score(flat, name)
            if hscore <= 0.02:
                continue
            # interpret as interleaved IQ
            if name == "complex64":
                cplx = vals
            else:
                v = flat
                n2 = (len(v) // 2) * 2
                cplx = v[0:n2:2] + 1j * v[1:n2:2]
            sym = _spectral_symmetry(cplx)
            structure = _spectral_structure(cplx)
            results.append({
                "dtype": name, "endian": endian, "complex": True,
                "histogram_score": round(hscore, 3),
                "spectral_symmetry": round(sym, 3),
                "spectral_structure": round(structure, 3),
                "confidence": round(hscore * (0.35 + 0.65 * structure), 3),
                "explanation": f"{name} {endian}-endian interleaved IQ: "
                               f"histogram {hscore:.2f}, spectral structure "
                               f"{structure:.2f}, symmetry {sym:.2f}",
            })
    results.sort(key=lambda r: -r["confidence"])
    if results:
        # normalise confidences to a distribution-ish scale
        rel = [r["confidence"] for r in results]
        s = sum(rel) or 1.0
        for r in results:
            r["confidence"] = round(r["confidence"] / s, 3)
    return results
