"""Stage S1: DC offset removal, I/Q imbalance correction (Gram-Schmidt),
clipping and dead-air detection. Non-destructive: returns a new array plus
a report; the raw recording is never modified."""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

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


def _to_analytic(x: np.ndarray) -> np.ndarray:
    """Hilbert transform of a real-valued recording, in blocks.

    A mono WAV from an SSB or audio receiver is real, so its spectrum is
    conjugate symmetric: every emission appears twice, mirrored about DC.
    A detector run on that sees a band twice as wide as reality and, once
    the two halves touch, one segment covering everything - which is
    precisely how the AO-73 capture produced a 16.4 kHz "signal" for a
    2 kHz carrier (report §10.2).  Converting to the analytic signal
    removes the mirror before anything downstream looks at the spectrum.

    scipy's ``hilbert`` is O(N log N) but allocates several float64 copies
    of the whole array, so long recordings are transformed in overlapping
    blocks and cross-faded, which keeps peak memory bounded and leaves no
    seam at the joins.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    block = 1 << 20
    if n <= block:
        return sig.hilbert(x).astype(np.complex64)
    overlap = 1 << 14
    out = np.empty(n, dtype=np.complex64)
    pos = 0
    while pos < n:
        a = max(0, pos - overlap)
        b = min(n, pos + block + overlap)
        seg = sig.hilbert(x[a:b])
        lo, hi = pos - a, min(pos + block, n) - a
        out[pos:pos + (hi - lo)] = seg[lo:hi].astype(np.complex64)
        pos += block
    return out


def condition(samples: np.ndarray, correct_iq: bool = True,
              max_analysis: int = 1 << 22,
              real_signal: bool = False) -> tuple:
    """Returns (conditioned complex64 array, ConditioningReport).

    ``real_signal`` marks a recording that carries a single real-valued
    channel; it is converted to the analytic signal first so that every
    later stage sees one copy of each emission instead of a mirrored
    pair.
    """
    rep = ConditioningReport()
    x = np.asarray(samples[:max_analysis], dtype=np.complex64) if \
        len(samples) > max_analysis else np.asarray(samples, dtype=np.complex64)
    if real_signal or (len(x) and not np.any(x.imag)):
        work = _to_analytic(x.real)
        rep.analytic_conversion = True
        rep.warnings.append(
            "real-valued input: converted to the analytic signal; only the "
            "positive half-band carries independent information")
        x = work
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
