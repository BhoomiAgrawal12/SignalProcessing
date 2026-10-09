"""Stage S1: DC offset removal, I/Q imbalance measurement, clipping and
dead-air detection. Non-destructive: returns a new array plus a report;
the raw recording is never modified.

Gram-Schmidt I/Q correction is opt-in (correct_iq=True): it assumes a
circular signal, but a non-rotating (zero-CFO) BPSK/ASK burst, or a short
framed burst whose repeated fields bias the I/Q statistics, reads as
imbalance, and "correcting" it broke classification. The imbalance is
always measured and reported. web/analyze.js does no correction either."""
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


def _hardware_dc(x: np.ndarray, block: int = 1024) -> tuple:
    """Receiver DC offset (LO leakage) is in every sample, noise-only
    stretches included; a signal's own mean (constant frame fields at zero
    CFO do not average out) exists only inside the burst. So the offset is
    measured on blocks >= 6 dB below the 90th-percentile block power (a
    burst's level whether it fills 10% or 90% of the file). Returns
    (dc, measured_on_quiet_blocks); without >= 2 quiet blocks it falls
    back to the whole-recording mean."""
    n = len(x) // block
    if n >= 4:
        blocks = x[:n * block].reshape(n, block)
        p = (np.abs(blocks) ** 2).mean(axis=1)
        quiet = p < 0.25 * np.percentile(p, 90)
        if quiet.sum() >= 2:
            return complex(blocks[quiet].mean()), True
    return complex(x.mean()), False


def condition(samples: np.ndarray, correct_iq: bool = False,
              max_analysis: int = 1 << 22) -> tuple:
    """Returns (conditioned complex64 array, ConditioningReport)."""
    rep = ConditioningReport()
    x = np.asarray(samples[:max_analysis], dtype=np.complex64)
    rep.analysed_samples = len(x)
    if len(samples) > max_analysis:
        # ponytail: first-window only, add --offset/sliding windows if
        # off-air captures put the burst late
        rep.warnings.append(
            f"only the first {max_analysis:,} of {len(samples):,} samples "
            f"({100 * max_analysis / len(samples):.1f}%) are analysed; "
            "signals after that point are not detected")
    work = np.array(x, dtype=np.complex64, copy=True)

    # --- DC offset ---------------------------------------------------------
    dc, from_quiet = _hardware_dc(work)
    rep.dc_offset = dc
    mag = float(np.abs(work).std() + 1e-12)
    if abs(dc) > 1e-4 * mag:
        work -= np.complex64(dc)
        rep.dc_removed = True
        if not from_quiet:
            rep.warnings.append(
                "no signal-free stretch to estimate the DC offset from: the "
                "whole-recording mean was removed, which also removes any "
                "mean the signal itself has (a zero-CFO burst with constant "
                "fields can be distorted)")

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
                "noise floor (dead air); those segments are still included "
                "in every later statistic")

    # --- I/Q imbalance: measured always, Gram-Schmidt only on request -------
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
