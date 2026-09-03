"""Stage S1: DC offset removal, I/Q imbalance correction (Gram-Schmidt),
clipping and dead-air detection. Non-destructive: returns a new array plus
a report; the raw recording is never modified."""
from __future__ import annotations

import numpy as np

from ..common.models import ConditioningReport


def _image_rejection_db(x: np.ndarray) -> float:
    """Ratio of strongest positive-frequency peak to its mirror."""
    n = min(len(x), 1 << 17)
    if n < 1024:
        return 0.0
    X = np.fft.fftshift(np.abs(np.fft.fft(x[:n] * np.hanning(n))) ** 2)
    half = n // 2
    pos = X[half + 1:]
    neg = X[:half][::-1]
    k = int(np.argmax(pos))
    if k >= len(neg) or pos[k] <= 0:
        return 0.0
    return float(10 * np.log10((pos[k] + 1e-20) / (neg[k] + 1e-20)))


def condition(samples: np.ndarray, correct_iq: bool = True,
              max_analysis: int = 1 << 22) -> tuple:
    """Returns (conditioned complex64 array, ConditioningReport)."""
    rep = ConditioningReport()
    x = np.asarray(samples[:max_analysis], dtype=np.complex64) if \
        len(samples) > max_analysis else np.asarray(samples, dtype=np.complex64)
    work = np.array(x, dtype=np.complex64, copy=True)

    # --- DC offset ---------------------------------------------------------
    dc = complex(work.mean())
    rep.dc_offset = dc
    mag = float(np.abs(work).std() + 1e-12)
    if abs(dc) > 1e-4 * mag:
        work -= np.complex64(dc)
        rep.dc_removed = True

    # --- clipping ----------------------------------------------------------
    peak = float(max(np.abs(work.real).max(), np.abs(work.imag).max()) or 1.0)
    clip_lvl = 0.985 * peak
    clipped = ((np.abs(work.real) >= clip_lvl) | (np.abs(work.imag) >= clip_lvl))
    rep.clipping_fraction = float(clipped.mean())
    if rep.clipping_fraction > 1e-3:
        rep.warnings.append(
            f"{rep.clipping_fraction*100:.2f}% of samples at full scale: "
            "recording appears clipped; spurious harmonics likely")

    # --- dead air ----------------------------------------------------------
    frame = 4096
    n_frames = len(work) // frame
    if n_frames >= 4:
        p = (np.abs(work[:n_frames * frame]) ** 2).reshape(n_frames, frame).mean(axis=1)
        floor = np.median(p)
        dead = p < 0.05 * floor
        rep.dead_air_fraction = float(dead.mean())
        if rep.dead_air_fraction > 0.3:
            rep.warnings.append(
                f"{rep.dead_air_fraction*100:.0f}% of recording is below the "
                "noise floor (dead air); statistics exclude those segments")

    # --- I/Q imbalance (Gram-Schmidt orthogonalisation) ----------------------
    i, q = work.real.astype(np.float64), work.imag.astype(np.float64)
    pi, pq = i @ i, q @ q
    if pq > 0 and pi > 0:
        rep.iq_gain_imbalance_db = float(10 * np.log10(pi / pq))
        cross = (i @ q) / np.sqrt(pi * pq)
        rep.iq_phase_error_deg = float(np.degrees(np.arcsin(np.clip(cross, -1, 1))))
        rep.image_rejection_before_db = _image_rejection_db(work)
        if correct_iq and (abs(rep.iq_gain_imbalance_db) > 0.05 or
                           abs(rep.iq_phase_error_deg) > 0.5):
            # Gram-Schmidt: make q orthogonal to i, equalise power
            q2 = q - (i @ q) / pi * i
            q2 *= np.sqrt(pi / (q2 @ q2))
            work = (i + 1j * q2).astype(np.complex64)
            rep.iq_corrected = True
            rep.image_rejection_after_db = _image_rejection_db(work)

    # --- normalise ----------------------------------------------------------
    rms = float(np.sqrt((np.abs(work) ** 2).mean()) or 1.0)
    work /= np.float32(rms)
    return work, rep
