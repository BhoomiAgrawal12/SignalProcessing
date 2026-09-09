"""Stage S3: frequency-shift a signal of interest to baseband, low-pass
filter and decimate. SciPy polyphase resampling (correct + fast enough for
offline files; a liquid-dsp fast path can be slotted behind this interface
later without touching callers)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal as sig


def channelize(x: np.ndarray, segment, target_oversample: float = 8.0,
               max_samples: int = 1 << 23, guard: float = 2.0,
               max_samples_per_symbol: float = 64.0) -> dict:
    """Extract ``segment`` from wideband stream ``x``.

    Three operations, each recorded so later measurements can be mapped
    back to absolute Hz: frequency shift to zero, CHANNEL FILTER, and
    decimation to about ``target_oversample`` samples per symbol-ish
    bandwidth.

    The channel filter is not cosmetic.  Without it the only band
    limiting is whatever the decimator's anti-alias filter happens to
    provide, which is a low-pass at the new Nyquist rather than at the
    signal's edges.  When the decimation ratio is small - a narrowband
    emission inside a much wider capture, e.g. a 2 kHz carrier in 48 kHz
    audio - neighbouring emissions survive into the "channelised" signal
    and every S4 estimate is then measured on the wrong thing: on the
    AO-73 capture the occupied-bandwidth figure came out three times too
    large, which pushed the symbol-rate search floor above the true
    1200 baud line and made the true rate unreachable.

    ``guard`` sets the filter's half-width as a multiple of the detected
    box's half-width.  It has to be generous: a detector box tracks the
    -8 dB points, so the pulse-shaping skirts already sit outside it, and
    those skirts carry BOTH the cyclostationary symbol-rate line and the
    raised-cosine transition the roll-off estimator measures.  Cutting
    into them costs the symbol rate and the roll-off; 2.0 leaves them
    intact while still rejecting anything a full box-width away.
    """
    s = np.asarray(x[segment.start_sample: segment.end_sample][:max_samples],
                   dtype=np.complex64)
    fc = segment.center_norm
    bw = max(segment.bandwidth_norm, 1e-4)

    # frequency shift to zero
    n = np.arange(len(s), dtype=np.float64)
    s = (s * np.exp(-2j * np.pi * fc * n)).astype(np.complex64)

    # channel filter: half-width = guard * (bw / 2), capped just under
    # the current Nyquist so a full-band segment is a no-op
    half = min(float(guard) * bw / 2.0, 0.49)
    filtered = False
    if half < 0.45 and len(s) > 256:
        # transition width of a quarter of the passband keeps the tap
        # count bounded while leaving the passband flat
        trans = max(half * 0.15, 1e-3)
        ntaps = int(np.clip(4.0 / trans, 31, 2047)) | 1
        ntaps = min(ntaps, (len(s) // 2) | 1)
        if ntaps >= 31:
            taps = sig.firwin(ntaps, 2.0 * half, window="hamming")
            s = sig.fftconvolve(s, taps.astype(np.float32),
                                mode="same").astype(np.complex64)
            filtered = True

    # decimate so that the occupied bandwidth is ~1/target_oversample of the
    # new rate (leaves room for pulse shaping skirts)
    decim_target = max(1.0, 1.0 / (bw * target_oversample))
    # never decimate so far that a symbol lands on fewer than 2 samples,
    # and never keep so many samples per symbol that the receiver's
    # interpolators run out of range
    frac = Fraction(1, 1) / Fraction(decim_target).limit_denominator(64)
    up, down = frac.numerator, frac.denominator
    if down > 1 or up > 1:
        s = sig.resample_poly(s, up, down).astype(np.complex64)
    new_rate_norm = up / down            # new_rate = original * up/down

    # trim leading/trailing noise: the detector's time resolution is one
    # waterfall row, so the box usually includes noise-only padding that
    # would poison EVM and bit alignment downstream
    s, trim = _trim_burst(s)

    return {"samples": s, "trimmed": trim,
            "freq_shift_norm": fc,
            "resample_up": up, "resample_down": down,
            "rate_ratio": new_rate_norm,
            "channel_filter_applied": filtered,
            "channel_filter_half_width_norm": (float(half) if filtered
                                               else None),
            # full passband width at the NEW rate: what S4 may treat as
            # noise-bearing.  1.0 (the whole band) when unfiltered.
            "analysis_band_norm": (min(1.0, 2.0 * half / new_rate_norm)
                                   if filtered else 1.0),
            "bandwidth_norm_at_new_rate": min(0.95, bw / new_rate_norm),
            "sample_rate": (segment.sample_rate * new_rate_norm
                            if segment.sample_rate else None)}


def _trim_burst(s: np.ndarray, guard: int = 32) -> tuple:
    """Keep the contiguous high-envelope region of the extracted channel.

    The boundary is taken at half the burst's own mean power (-3 dB) and
    then moved INWARD by a small guard.  The shipped version used a -6 dB
    boundary and moved outward by 32 samples, on the reasoning that the
    detector's time resolution is coarse and a margin avoids clipping the
    burst.  The cost of that margin turned out to be severe: a handful of
    noise-only samples at the tail of an otherwise clean 32PSK record
    took its EVM from 1.1% to 7.6%, because every feedforward estimator
    in S6 has its weakest support exactly there.  Losing a few symbols at
    each end is far cheaper than keeping a few samples of noise.
    """
    if len(s) < 1024:
        return s, (0, len(s))
    p = np.abs(s) ** 2
    win = max(64, len(s) // 512)
    k = np.ones(win) / win
    sm = np.convolve(p, k, mode="same")
    hi = np.median(sm[sm > np.median(sm)])
    thr = 0.5 * hi
    idx = np.nonzero(sm > thr)[0]
    if idx.size == 0:
        return s, (0, len(s))
    a = int(idx[0]) + guard
    b = int(idx[-1]) - guard
    if b - a < max(256, len(s) // 8):        # burst too short to guard
        a, b = int(idx[0]), int(idx[-1]) + 1
    return s[a:b], (a, b)
