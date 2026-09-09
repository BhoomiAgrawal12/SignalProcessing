"""Stage S4: physical parameter estimation.

Tiered strategy (report §13): cheap estimators first (spectral line, OBW,
M2M4), then a time-smoothed cyclic periodogram sweep for symbol rate
confirmation. Every estimate carries a confidence and its method name.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

from ..common.models import (SignalParameters, Estimate, EstimateState,
                             Verdict)

# Signal kurtosis ka = E|s|^4 / (E|s|^2)^2 per constellation family.  The
# M2M4 estimator needs it; the shipped code accepted the argument and
# then ignored it, which is the whole of defect D1 (report §3.1).
# Values are computed from the unit-power constellation tables, not
# guessed - see ``signal_kurtosis``.
_KURTOSIS_CACHE: dict = {}


def signal_kurtosis(modulation: str) -> float:
    """ka = E|s|^4 / (E|s|^2)^2 for one modulation's own alphabet.

    Constant-modulus alphabets (PSK, FSK, GMSK, OQPSK) give exactly 1.0;
    16QAM gives 1.32, 64QAM 1.381, 256QAM 1.395 - the numbers whose
    mismatch produced the measured 6.7 / 5.6 dB SNR ceilings.
    """
    if modulation in _KURTOSIS_CACHE:
        return _KURTOSIS_CACHE[modulation]
    ka = 1.0
    try:
        from ..demod.constellations import CONSTELLATIONS
        table = np.asarray(CONSTELLATIONS[modulation][0], dtype=np.complex128)
        p = np.abs(table) ** 2
        ka = float((p ** 2).mean() / (p.mean() ** 2))
    except Exception:
        ka = 1.0
    _KURTOSIS_CACHE[modulation] = ka
    return ka


def m2m4_ceiling_db(ka_true: float, ka_assumed: float = 1.0) -> float:
    """Noise-free ceiling of an M2M4 estimator that assumes ``ka_assumed``
    on a signal whose true kurtosis is ``ka_true``.

    With no noise, M2 = S and M4 = ka_true * S^2, so the estimator solves
    S_est^2 = (2 M2^2 - M4) / (2 - ka_assumed) = S^2 (2 - ka_true) /
    (2 - ka_assumed) and reports SNR = S_est / (M2 - S_est).  For
    ka_assumed = 1 this evaluates to 6.72 dB at 16QAM and 5.67 dB at
    64QAM, matching the measured ceilings in report §3.2 to 0.1 dB.
    Returns ``inf`` when the assumption is exact (no ceiling).
    """
    r = (2.0 - ka_true) / (2.0 - ka_assumed)
    if r <= 0:
        return float("-inf")
    root = float(np.sqrt(r))
    if root >= 1.0:
        return float("inf")
    return float(10 * np.log10(root / (1.0 - root)))


# --------------------------------------------------------------------------
def occupied_bandwidth(x: np.ndarray, fraction: float = 0.99) -> dict:
    """ITU-R SM.443 99%-power bandwidth, -3 dB bandwidth and, when the
    symbol rate is known, the RRC excess bandwidth.

    The shipped version subtracted a noise floor and returned ``None``
    for everything when the subtraction emptied the spectrum, which is
    defect D4 (report §5: "occupied bandwidth is None ... output is
    documented but missing").  The floor subtraction is kept because it
    is what makes the 99% integral defensible at low SNR, but it now
    falls back to the unsubtracted spectrum instead of giving up, and
    the method actually used is reported.
    """
    n = min(len(x), 1 << 20)
    nper = min(8192, max(256, n // 8))
    nper = min(nper, n)
    f, p = sig.welch(x[:n], fs=1.0, nperseg=nper, return_onesided=False,
                     detrend=False)
    idx = np.argsort(f)
    f, p = f[idx], p[idx]
    # noise floor subtraction for a defensible OBW on noisy captures;
    # the floor tracks the bin-to-bin variability (few Welch averages
    # leave exponential noise residue above the bare median that would
    # otherwise dominate the 99% integral at low SNR)
    med = np.median(p)
    mad = np.median(np.abs(p - med))
    floor = med + 2.0 * 1.4826 * mad
    ps = np.clip(p - floor, 0, None)
    method = "cumulative PSD above the noise floor (ITU-R SM.443)"
    if ps.sum() <= 0:
        # every bin sat under the robust floor: the band is either empty
        # or completely filled by the signal.  Retreat to the bare
        # spectrum rather than reporting None for a measurable quantity.
        ps = np.clip(p - float(np.min(p)), 0, None)
        method = "cumulative PSD, bare spectrum (noise floor not separable)"
    total = float(ps.sum())
    if total <= 0:
        return {"obw99": None, "obw3db": None, "f_center": 0.0,
                "method": "no measurable power", "psd_freq": f,
                "psd": 10 * np.log10(p + 1e-20)}
    c = np.cumsum(ps) / total
    lo = f[np.searchsorted(c, (1 - fraction) / 2)]
    hi = f[min(np.searchsorted(c, 1 - (1 - fraction) / 2), len(f) - 1)]
    # smooth before the -3dB read: a single noisy bin must not define
    # the peak nor the crossing points
    win = max(3, (len(ps) // 256) | 1)
    ps_s = np.convolve(ps, np.ones(win) / win, mode="same")
    peak = ps_s.max()
    above = f[ps_s > peak / 2]
    obw3 = float(above[-1] - above[0]) if len(above) else None
    centroid = float((f * ps).sum() / total)
    return {"obw99": float(hi - lo), "obw3db": obw3, "f_center": centroid,
            "method": method, "psd_freq": f, "psd_lin": ps_s,
            "psd": 10 * np.log10(p + 1e-20)}


def _rolloff_psd(x: np.ndarray, symbol_rate: float,
                 min_bins_across_transition: int = 12) -> tuple:
    """A PSD chosen for measuring a roll-off, not for measuring a band.

    The occupied-bandwidth PSD maximises frequency resolution, which is
    the wrong trade here: the transition is 0.35*Rs wide and needs only a
    dozen bins to describe, while the fit needs a SMOOTH curve, and every
    halving of the segment length doubles the number of Welch averages.
    Measured on the same captures, taking that trade turns a fit residual
    of 0.19 into one of 0.06 and a roll-off of 0.85 into 0.36.
    """
    n = len(x)
    if n < 2048 or not symbol_rate:
        return None, None
    # a narrow roll-off (b = 0.15) spans 0.15*Rs; keep at least
    # min_bins_across_transition bins inside it
    want = min_bins_across_transition / max(0.15 * symbol_rate, 1e-6)
    nper = int(2 ** round(np.log2(np.clip(want, 128, 8192))))
    nper = int(min(nper, max(128, n // 24)))     # at least ~48 averages
    if nper < 128:
        return None, None
    f, p = sig.welch(x, fs=1.0, nperseg=nper, noverlap=nper // 2,
                     return_onesided=False, detrend=False)
    order = np.argsort(f)
    f, p = f[order], np.asarray(p[order], dtype=np.float64)
    med = np.median(p)
    mad = np.median(np.abs(p - med))
    p = np.clip(p - (med + 2.0 * 1.4826 * mad), 0.0, None)
    return f, p


def excess_bandwidth(psd_freq: np.ndarray, psd_lin: np.ndarray,
                     symbol_rate: float, f_center: float = 0.0,
                     smooth_fraction: float = 0.03) -> dict:
    """Root-raised-cosine roll-off, by fitting the transition shape.

    The shipped estimate was ``obw99 / Rs - 1``, which measures where the
    99% power integral stops rather than where the filter rolls off; it
    read 0.14 for a true roll-off of 0.35 (defect D3, report §5).

    The transmitted PSD of an RRC-shaped signal is the raised cosine, so
    its shape is known exactly:

        P(f) = 1                                     |f| <= (1-b)Rs/2
             = 0.5*(1 + cos(pi/(b*Rs) * (|f| - (1-b)Rs/2)))   transition
             = 0                                     |f| >= (1+b)Rs/2

    Reading two threshold crossings off that curve is the obvious method
    and it does not survive contact with real captures: the 10% crossing
    sits in the tail, where a 28 dB noise floor keeps the curve above
    threshold well past the true stop, and the 90% crossing sits in the
    passband, where a few percent of ripple from a short record moves it
    almost anywhere.  Both failures were measured, inflating a true 0.35
    to 0.64 and to 0.79 respectively.

    Fitting the whole transition instead uses every point in it, so
    ripple averages out and tail noise contributes little (the model is
    zero there, and so is most of the data).  One free parameter, a
    one-dimensional search.

    Accuracy, measured through the whole pipeline on 60-frame bursts at
    each family's working SNR: about +-0.15 on a true 0.35, against the
    shipped estimator's systematic 60% underestimate.  ``fit_residual``
    is returned so a caller can see how well the model described the
    data, and a fit that plainly does not is reported as no measurement
    at all rather than as a number.
    """
    if symbol_rate is None or symbol_rate <= 0 or psd_lin is None:
        return {"rolloff": None, "method": "symbol rate unknown"}
    f = np.asarray(psd_freq, dtype=np.float64) - float(f_center)
    p = np.asarray(psd_lin, dtype=np.float64)
    if len(f) < 32:
        return {"rolloff": None, "method": "spectrum too coarse"}
    # flat-top reference: the median of the passband interior, which
    # exists for every roll-off below 1.0
    flat = np.abs(f) < 0.3 * symbol_rate
    if flat.sum() < 4:
        return {"rolloff": None, "method": "passband too narrow to sample"}
    ref = float(np.median(p[flat]))
    if ref <= 0:
        return {"rolloff": None, "method": "no passband power"}
    # Framed traffic is periodic, so its spectrum is a comb whose ENVELOPE
    # is the raised cosine.  Fitting the comb itself makes the residual
    # large enough to reject a perfectly good spectrum, so the curve is
    # smoothed over a window narrow enough to preserve the narrowest
    # roll-off worth reporting (b = 0.1 spans 0.1*Rs) and wide enough to
    # bridge the lines.
    bin_width = float(np.median(np.diff(f))) if len(f) > 1 else 0.0
    if bin_width > 0:
        win = int(np.clip(round(smooth_fraction * symbol_rate /
                                bin_width), 1, 401))
        if win > 2:
            if win % 2 == 0:
                win += 1
            kern = np.ones(win) / win
            p = np.convolve(p, kern, mode="same") / np.convolve(
                np.ones_like(p), kern, mode="same")
            ref = float(np.median(p[flat]))
    norm = np.clip(p / max(ref, 1e-30), 0.0, 1.5)

    # fit over the region that can contain a transition for any b <= 1:
    # from (1-1)Rs/2 = 0 out to (1+1)Rs/2 = Rs.  Start at 0.25*Rs so a
    # mis-set flat-top reference cannot dominate the fit.
    band = (np.abs(f) >= 0.25 * symbol_rate) & \
           (np.abs(f) <= 1.05 * symbol_rate)
    if band.sum() < 16:
        return {"rolloff": None,
                "method": "raised-cosine transition not observable"}
    xf, yf = np.abs(f[band]), norm[band]

    def _model(beta):
        lo = (1.0 - beta) * symbol_rate / 2.0
        hi = (1.0 + beta) * symbol_rate / 2.0
        out = np.zeros_like(xf)
        out[xf <= lo] = 1.0
        mid = (xf > lo) & (xf < hi)
        if beta > 0:
            out[mid] = 0.5 * (1.0 + np.cos(
                np.pi / (beta * symbol_rate) * (xf[mid] - lo)))
        return out

    grid = np.linspace(0.02, 1.0, 99)
    errs = np.array([float(np.mean((yf - _model(b)) ** 2)) for b in grid])
    k = int(np.argmin(errs))
    beta = float(grid[k])
    # parabolic refinement between neighbouring grid points
    if 0 < k < len(grid) - 1:
        y0, y1, y2 = errs[k - 1], errs[k], errs[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-30:
            beta += float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0)) * \
                (grid[1] - grid[0])
    residual = float(np.sqrt(errs[k]))
    # A fit that does not describe the data is not a measurement.  The
    # residual is in units of the normalised PSD, so 0.25 means the model
    # is a quarter of full scale away from the data on average - that is
    # not a raised cosine.
    if residual > 0.25:
        return {"rolloff": None,
                "fit_residual": round(residual, 4),
                "method": "spectrum does not fit a raised-cosine shape"}
    return {"rolloff": round(float(np.clip(beta, 0.0, 1.0)), 3),
            "fit_residual": round(residual, 4),
            "n_points": int(band.sum()),
            "method": "raised-cosine transition shape fit"}


def carrier_line(x: np.ndarray, orders=(2, 4), max_samples: int = 1 << 18) -> dict:
    """Independent carrier location from the strongest M-power line.

    Report §10.5 item 3: on the AO-73 capture the x^2 line put the
    carrier at 1102 Hz while S4's own answer was 6 Hz, and nothing
    reconciled the two.  This returns the line so the pipeline can
    cross-check - it is deliberately kept separate from
    :func:`carrier_offset`, which feeds the demodulator, so a
    disagreement stays visible instead of being averaged away.
    """
    n = min(len(x), max_samples)
    if n < 1024:
        return {"carrier_norm": None, "quality": 0.0, "order": 0}
    xx = np.asarray(x[:n], dtype=np.complex128)
    best = {"carrier_norm": None, "quality": 0.0, "order": 0}
    freqs = np.fft.fftfreq(n)
    for M in orders:
        z = (xx ** M) * np.hanning(n)
        Z = np.abs(np.fft.fft(z))
        Z[0] = 0.0
        med = float(np.median(Z)) + 1e-30
        k = int(np.argmax(Z))
        q = float(Z[k] / med)
        if q > best["quality"]:
            best = {"carrier_norm": float(freqs[k] / M), "quality": q,
                    "order": M, "line_norm": float(freqs[k])}
    return best


def _m2m4_core(x: np.ndarray, ka: float) -> tuple:
    """Solve the M2/M4 pair for (signal power, noise power).

    For a signal of power S with kurtosis ka in complex Gaussian noise of
    power N (kw = 2):
        M2 = S + N
        M4 = ka*S^2 + 4*S*N + 2*N^2 = (ka - 2)*S^2 + 2*M2^2
    hence S^2 = (2*M2^2 - M4) / (2 - ka).  Returns (S, N, M2) with S = nan
    when the moments are inconsistent with the model.
    """
    m2 = float((np.abs(x) ** 2).mean())
    m4 = float((np.abs(x) ** 4).mean())
    denom = 2.0 - float(ka)
    if abs(denom) < 1e-6:
        return float("nan"), float("nan"), m2
    inner = (2.0 * m2 * m2 - m4) / denom
    if inner <= 0:
        return float("nan"), float("nan"), m2
    s_pow = float(np.sqrt(inner))
    return s_pow, m2 - s_pow, m2


def spectral_snr(x: np.ndarray, nper: int = 4096,
                 max_samples: int = 1 << 20,
                 reference_bandwidth: float = None,
                 band_limit: float = None) -> Estimate:
    """SNR from the out-of-band noise floor.

    Completely independent of the constellation: the noise power spectral
    density is read from the quiet part of the band, scaled to the full
    band, and subtracted from the total power.  This is what an analyst
    does with a spectrum display, and unlike M2M4 it does not care
    whether the signal is constant modulus - which is exactly why it can
    arbitrate the M2M4 saturation of report §3.2.

    It needs a noise-only region to exist.  A channelised signal at the
    pipeline's 8x target oversampling occupies about a sixth of the band,
    so the region is normally there; when the signal fills the band the
    estimate is returned as ``unresolvable`` rather than guessed.

    ``reference_bandwidth`` (normalised, cycles/sample) selects the
    bandwidth the noise is counted in.  Passing the symbol rate yields
    Es/N0 - the standard link quantity, the one that predicts BER, the
    one the synchronised EVM measures, and the only one that does not
    change when the channeliser decimates.  Omitting it yields the
    full-band SNR at the current rate, which is what the shipped
    estimator reported.
    """
    n = min(len(x), max_samples)
    if n < 1024:
        return Estimate(value=None, confidence=0.0,
                        method="out-of-band noise floor",
                        state=EstimateState.UNRESOLVABLE,
                        verdict=Verdict.UNKNOWN,
                        detail={"reason": "too few samples"})
    nper = int(min(nper, n // 4))
    _f, psd = sig.welch(np.asarray(x[:n]), fs=1.0, nperseg=nper,
                        noverlap=nper // 2, return_onesided=False,
                        detrend=False)
    psd = np.asarray(psd, dtype=np.float64)
    # restrict every statistic to the band the recording actually
    # occupies; a channel filter's empty stopband is not noise-only data
    # Without an explicit band limit the whole band is assumed to carry
    # receiver noise, which is true for any unfiltered capture.  The
    # channeliser passes its passband width when it has applied a channel
    # filter, because a filter stopband is EMPTY rather than noise-only
    # and counting it as noise drives the density estimate to zero.
    half = float(band_limit) if band_limit else 0.5
    half = float(np.clip(half, 1e-4, 0.5))
    inband = np.abs(_f) <= half
    if inband.sum() < 16:
        inband = np.ones(len(psd), dtype=bool)
        half = 0.5
    psd = psd[inband]
    band_width = 2.0 * half
    # Noise density: a low quantile of the bin powers, de-biased for the
    # chi-square residue that Welch averaging leaves behind.  The
    # quantile ADAPTS to how much of the band the signal covers: a
    # channel-filtered narrowband capture can leave only a sixth of the
    # bins noise-only, and a fixed 25th percentile would then be measured
    # on the signal itself.  Two refinement passes are enough because the
    # occupancy estimate only has to be right to within a factor of two.
    n_avg = max(1, 2 * n // nper - 1)
    q = 0.25
    dens = float(np.quantile(psd, q)) / _chi2_quantile_bias(n_avg, q)
    occupancy = float((psd > 3.0 * dens).mean())
    for _ in range(2):
        if occupancy <= 0.5:
            break
        q = float(np.clip(0.5 * (1.0 - occupancy), 0.02, 0.25))
        dens = float(np.quantile(psd, q)) / _chi2_quantile_bias(n_avg, q)
        occupancy = float((psd > 3.0 * dens).mean())
    total = float(psd.sum())
    excess = float(np.clip(psd - dens, 0.0, None).sum())
    noise_total = dens * len(psd)          # power in the analysed band
    if occupancy > 0.9 or noise_total <= 0:
        return Estimate(value=None, confidence=0.0,
                        method="out-of-band noise floor",
                        state=EstimateState.UNRESOLVABLE,
                        verdict=Verdict.UNKNOWN,
                        detail={"reason": "no noise-only region in the band "
                                          f"(occupancy {occupancy:.2f})",
                                "occupancy": round(occupancy, 3)})
    if excess <= 0.05 * total:
        # Either the band is empty or the signal fills it uniformly.  In
        # both cases the "floor" this estimator found IS the signal, and
        # the number it would report is meaningless - a Nyquist-rate QPSK
        # stream at 30 dB comes out as -13.8 dB if this is not caught.
        return Estimate(value=None, confidence=0.0,
                        method="out-of-band noise floor",
                        state=EstimateState.UNRESOLVABLE,
                        verdict=Verdict.UNKNOWN,
                        detail={"reason": "no power separable from the "
                                          "noise floor: the band is either "
                                          "empty or uniformly filled",
                                "excess_fraction": round(
                                    excess / max(total, 1e-30), 4),
                                "occupancy": round(occupancy, 3)})
    full_band = float(10 * np.log10(excess / noise_total))
    detail = {"occupancy": round(occupancy, 3),
              "noise_density": float(dens),
              "total_power": total,
              "welch_averages": int(n_avg),
              "noise_quantile": round(q, 3),
              "analysis_band_norm": round(band_width, 5),
              "full_band_snr_db": round(full_band, 1)}
    snr, method = full_band, "out-of-band noise floor (full-band SNR)"
    if reference_bandwidth and reference_bandwidth > 0:
        # noise in ``reference_bandwidth`` cycles/sample: the density is
        # per analysed-band bin, so scale by the ratio of bandwidths
        ref_noise = noise_total * float(reference_bandwidth) / band_width
        if ref_noise > 0:
            snr = float(10 * np.log10(excess / ref_noise))
            method = ("out-of-band noise floor, referred to the symbol "
                      "rate (Es/N0)")
            detail["reference_bandwidth_norm"] = float(reference_bandwidth)
    return Estimate(value=round(snr, 1), confidence=0.85, method=method,
                    state=EstimateState.VALID, detail=detail)


def _chi2_quantile_bias(n_avg: int, q: float) -> float:
    """E[q-quantile of a chi2_{2k}/(2k) variate], the factor by which a
    quantile of Welch bin powers under-reads the true mean density."""
    from scipy import stats
    return float(stats.chi2.ppf(q, 2 * n_avg) / (2 * n_avg))


def m2m4_snr(x: np.ndarray, kurtosis_signal: float = 1.0,
             max_samples: int = 1 << 20,
             reference_snr_db: float = None,
             reference_margin_db: float = 3.0) -> Estimate:
    """Moment-based blind SNR estimate with an explicit validity state.

    The shipped estimator hard-coded the constant-modulus (ka = 1) form
    and silently saturated on QAM/APSK - 16QAM read 6.7 dB at every true
    SNR above about 15 dB (report §3.2).  Two changes fix that:

    1. ``kurtosis_signal`` is actually used, through the general
       Pauluzzi-Beaulieu relation S^2 = (2 M2^2 - M4) / (2 - ka).  Callers
       that know the modulation (S5 and S6 do) get an unbiased estimate;
       :func:`signal_kurtosis` computes ka from the constellation table.
    2. Saturation is *detected* rather than assumed away.  The ceiling of
       a ka-mismatched M2M4 estimator is analytic
       (:func:`m2m4_ceiling_db`) but depends on the unknown true
       kurtosis, so the detector uses an independent, kurtosis-free
       reference - the out-of-band noise floor of :func:`spectral_snr`.
       When the moment estimate falls more than ``reference_margin_db``
       below that reference, the value is a floor, not a measurement.

    Returns an :class:`Estimate`; ``state`` is ``saturated`` when the
    value is only a lower bound and ``unresolvable`` when the moments do
    not fit the signal-plus-noise model at all.
    """
    x = np.asarray(x[:max_samples])
    ka = float(kurtosis_signal)
    method = (f"M2M4 moment estimator (ka={ka:.3f})" if abs(ka - 1.0) > 1e-9
              else "M2M4 moment estimator (constant-modulus assumption)")
    s_pow, n_pow, m2 = _m2m4_core(x, ka)
    if not np.isfinite(s_pow):
        return Estimate(value=None, confidence=0.0, method=method,
                        state=EstimateState.UNRESOLVABLE,
                        verdict=Verdict.UNKNOWN,
                        detail={"reason": "M2/M4 inconsistent with the "
                                          "signal-plus-Gaussian-noise model",
                                "kurtosis_assumed": round(ka, 4)})
    detail = {"kurtosis_assumed": round(ka, 4)}
    if n_pow <= 1e-12 * m2:
        detail.update(reason="no measurable noise power",
                      reported_as=">= 40.0 dB")
        return Estimate(value=40.0, confidence=0.3, method=method,
                        state=EstimateState.SATURATED, detail=detail)
    snr = round(float(10 * np.log10(s_pow / n_pow)), 1)

    state = EstimateState.VALID
    if reference_snr_db is not None:
        detail["independent_reference_db"] = round(float(reference_snr_db), 1)
        if snr < float(reference_snr_db) - float(reference_margin_db):
            state = EstimateState.SATURATED
            detail["reason"] = (
                f"the out-of-band noise floor puts the SNR at "
                f"{reference_snr_db:.1f} dB while the moment estimator "
                f"reports {snr:.1f} dB: the moment value is a floor")
            detail["reported_as"] = f">= {snr} dB"
            if abs(ka - 1.0) < 1e-9:
                detail["hint"] = (
                    "constant-modulus assumption applied to a "
                    "non-constant-modulus signal; supply the modulation "
                    "kurtosis or use the EVM-based SNR after "
                    "synchronisation")
    conf = 0.75 if state == EstimateState.VALID else 0.2
    return Estimate(value=snr, confidence=conf, method=method,
                    state=state, detail=detail)


def snr_from_evm(evm_percent: float) -> Estimate:
    """Es/N0 implied by the post-synchronisation error-vector magnitude.

    This is the estimator to fall back on for dense QAM/APSK, where the
    moment estimator saturates (report §3.4).  EVM is measured on
    matched-filtered, symbol-spaced samples, so 1/EVM^2 is exactly Es/N0
    - the same quantity :func:`spectral_snr` reports when it is given the
    symbol rate, and the same one the pipeline publishes as ``snr_db``.
    Keeping the two in one unit is what makes the S4/S6 reconciliation a
    comparison instead of a unit conversion.
    """
    if not evm_percent or evm_percent <= 0:
        return Estimate(value=None, confidence=0.0,
                        method="EVM after synchronisation",
                        state=EstimateState.UNRESOLVABLE,
                        verdict=Verdict.UNKNOWN)
    es_n0 = float(-20 * np.log10(float(evm_percent) / 100.0))
    return Estimate(value=round(es_n0, 1), confidence=0.9,
                    method="EVM after synchronisation (Es/N0)",
                    state=EstimateState.VALID,
                    detail={"evm_percent": float(evm_percent)})


def carrier_offset(x: np.ndarray, orders=(2, 4, 8)) -> dict:
    """CFO from the strongest spectral line of x^M (classic M-power method).
    Also reports which order produced the line - a modulation-order hint."""
    n = min(len(x), 1 << 18)
    xx = x[:n] / (np.abs(x[:n]) + 1e-12)      # limiter improves the line
    best = {"cfo": 0.0, "order": 0, "line_snr": 0.0}
    for M in orders:
        y = xx ** M
        Y = np.abs(np.fft.fft(y * np.hanning(n)))
        Y[0] = 0
        k = int(np.argmax(Y))
        freqs = np.fft.fftfreq(n)
        med = np.median(Y)
        line_snr = float(Y[k] / (med + 1e-12))
        if line_snr > best["line_snr"]:
            best = {"cfo": float(freqs[k] / M), "order": M,
                    "line_snr": line_snr}
    best["confident"] = best["line_snr"] > 8.0
    return best


def symbol_rate(x: np.ndarray, min_rate: float = 1e-4,
                lags=(0, 1, 2, 4), obw99: float = None,
                obw_guard: float = 0.2, max_candidates: int = 8) -> dict:
    """Blind symbol-rate estimation via time-smoothed cyclic periodograms.

    The magnitude nonlinearity |x|^2 exposes a spectral line at the symbol
    rate for shaped linear modulations; summing cyclic periodograms of
    x(t)x*(t-d) over several delays d strengthens the line for
    low-excess-bandwidth signals. Returns candidates ranked by prominence.
    """
    n = min(len(x), 1 << 19)
    xx = np.asarray(x[:n], dtype=np.complex64)
    acc = None
    for d in lags:
        y = xx[d:] * np.conj(xx[:len(xx) - d]) if d else (np.abs(xx) ** 2).astype(np.complex64)
        m = len(y)
        Y = np.abs(np.fft.fft((y - y.mean()) * np.hanning(m), n))
        acc = Y if acc is None else acc[:len(Y)] + Y[:len(acc)]
    freqs = np.fft.fftfreq(len(acc))
    # the symbol rate of a linear modulation cannot sit far below its
    # occupied bandwidth; this guard rejects low-frequency envelope spurs
    # (prominent for QAM) without excluding real candidates
    # ``obw_guard`` * OBW99 is the floor: a shaped linear modulation's
    # rate cannot sit far below its occupied bandwidth.  0.2 rather than
    # 0.25 because an audio-rate telemetry signal at 40 samples/symbol
    # has its rate at OBW99/1.35 = 0.74*OBW99, and the old floor sat
    # uncomfortably close to that for narrow, heavily-filtered captures.
    floor = min_rate
    if obw99:
        floor = max(min_rate, obw_guard * obw99)
    pos = (freqs > floor) & (freqs < 0.5)
    f_pos, a_pos = freqs[pos], acc[pos]
    if len(a_pos) == 0:
        return {"candidates": [], "confidence": 0.0, "min_rate": floor}
    # peak prominence against a smoothed background; the median filter is
    # one-sided at the mask edge, which inflates prominence there and used
    # to elect a spurious "rate" exactly at 0.25*OBW - so the edge bins
    # are excluded from candidacy
    bg = sig.medfilt(a_pos, kernel_size=51)
    prom = a_pos / (bg + 1e-12)
    edge = min(26, len(prom) // 4)
    prom[:edge] = 0.0
    prom[-3:] = 0.0
    order = np.argsort(prom)[::-1]
    cands = []
    for k in order[:20]:
        f = float(f_pos[k])
        # skip harmonics and near-duplicates of an already-kept candidate.
        # (An explicit sub-harmonic promotion used to live here to undo
        # the estimator electing 2*Rs; the S3 channel filter removed the
        # cause - a truncated, aliased channel - and the promotion then
        # cost more than it bought, demoting a true 64QAM rate line to
        # its own half.  The runners-up remain available to the receiver
        # trials, which is the right place to resolve a rate ambiguity
        # because a lock is decisive and a periodogram line is not.)
        if any(abs(f - m * c["rate_norm"]) < 2.0 / n
               for c in cands for m in (1, 2, 3, 4)):
            continue
        cands.append({"rate_norm": f, "prominence": float(prom[k])})
        if len(cands) >= max_candidates:
            break
    conf = 0.0
    if cands:
        top = cands[0]["prominence"]
        conf = float(min(1.0, max(0.0, (top - 3) / 20)))
    return {"candidates": cands, "confidence": conf, "min_rate": floor}


def fsk_tones(x: np.ndarray, max_tones: int = 8) -> dict:
    """Instantaneous-frequency histogram: tone count, spacing, and whether
    the signal looks like FSK at all.

    The decisive FSK evidence is the constant envelope (measured on our
    synthetic corpus: RRC-shaped PSK/QAM cv >= 0.27, FSK cv <= 0.13 down to
    ~12 dB SNR). The tone histogram then only determines the order; its
    smoothing window is chosen adaptively because the symbol rate is not
    yet known at this stage.
    """
    n = min(len(x), 1 << 18)
    xx = x[:n]
    amp_all = np.abs(xx)
    cv = float(amp_all.std() / (amp_all.mean() + 1e-12))
    if cv > 0.17:
        return {"is_fsk": False, "amplitude_cv": cv}
    ph = np.unwrap(np.angle(xx))
    inst_raw = np.diff(ph) / (2 * np.pi)
    amp = amp_all[1:]
    good = amp > 0.3 * np.median(amp)
    inst_raw = inst_raw[good]
    if len(inst_raw) < 1000:
        return {"is_fsk": False, "amplitude_cv": cv}

    best = None
    for w in (2, 4, 8, 16, 32):
        inst_f = np.convolve(inst_raw, np.ones(w) / w, mode="valid")
        lo, hi = np.percentile(inst_f, [1, 99])
        span = hi - lo
        lo -= 0.15 * span
        hi += 0.15 * span
        hist, edges = np.histogram(inst_f, bins=256, range=(lo, hi))
        hist_s = sig.medfilt(hist.astype(float), 5)
        peaks, _ = sig.find_peaks(hist_s, height=0.25 * hist_s.max(),
                                  distance=8)
        if not (2 <= len(peaks) <= max_tones):
            continue
        width = max(2, len(hist_s) // (4 * len(peaks)))
        in_peak = np.zeros(len(hist_s), dtype=bool)
        for pk in peaks:
            in_peak[max(0, pk - width):pk + width] = True
        frac = float(hist_s[in_peak].sum() / hist_s.sum())
        centers = 0.5 * (edges[peaks] + edges[peaks + 1])
        cand = {"n_tones": int(len(peaks)), "frac": frac,
                "centers": centers, "hist": hist_s, "edges": edges, "w": w}
        if best is None or frac > best["frac"]:
            best = cand
    if best is None or best["n_tones"] not in (2, 4, 8) or best["frac"] < 0.5:
        return {"is_fsk": False, "amplitude_cv": cv,
                "note": "constant envelope but no clean tone structure"}
    centers = np.sort(best["centers"])
    diffs = np.diff(centers)
    return {"is_fsk": True, "tone_count": best["n_tones"],
            "amplitude_cv": cv,
            "tone_freqs_norm": [float(c) for c in centers],
            "deviation_norm": float(diffs.mean()) if len(diffs) else 0.0,
            "mass_fraction": best["frac"],
            "smoothing_window": best["w"],
            "histogram": best["hist"].tolist(),
            "bin_centers": (0.5 * (best["edges"][:-1] + best["edges"][1:])).tolist()}


def fsk_symbol_rate(x: np.ndarray, min_rate: float = 1e-4) -> dict:
    """Symbol rate of an FSK signal from the spectral line of the tone
    *transition* process |d/dt inst_freq|."""
    n = min(len(x), 1 << 18)
    inst_f = np.diff(np.unwrap(np.angle(x[:n]))) / (2 * np.pi)
    d = np.abs(np.diff(inst_f))
    d -= d.mean()
    D = np.abs(np.fft.fft(d * np.hanning(len(d))))
    freqs = np.fft.fftfreq(len(d))
    pos = (freqs > min_rate) & (freqs < 0.45)
    if not pos.any():
        return {"rate_norm": None, "confidence": 0.0, "min_rate": min_rate}
    f_pos, a_pos = freqs[pos], D[pos]
    bg = sig.medfilt(a_pos, kernel_size=51)
    prom = a_pos / (bg + 1e-12)
    # exclude the mask edges where the one-sided median filter inflates
    # prominence (same artifact as in symbol_rate)
    edge = min(26, len(prom) // 4)
    prom[:edge] = 0.0
    prom[-26:] = 0.0
    k = int(np.argmax(prom))
    # the transition process is an impulse train: harmonics can outscore
    # the fundamental after background normalisation, so prefer a
    # sub-harmonic when it carries a comparable line
    best_k = k
    for div in (4, 3, 2):
        f_sub = f_pos[k] / div
        j = int(np.argmin(np.abs(f_pos - f_sub)))
        lo, hi = max(0, j - 2), min(len(prom), j + 3)
        jj = lo + int(np.argmax(prom[lo:hi]))
        if prom[jj] > 0.4 * prom[k]:
            best_k = jj
            break
    conf = float(min(1.0, max(0.0, (prom[best_k] - 3) / 20)))
    if conf <= 0:
        return {"rate_norm": None, "confidence": 0.0}
    return {"rate_norm": float(f_pos[best_k]), "confidence": conf}


def ofdm_detect(x: np.ndarray, fft_sizes=(64, 128, 256, 512, 1024, 2048)) -> dict:
    """Multi-evidence OFDM detector.

    Cyclic-prefix correlation alone false-positives on oversampled
    single-carrier signals and slow FSK, so a detection requires ALL of:
    1. strong CP correlation (99th percentile of the lag-N product),
    2. a peaky (bursty) correlation profile,
    3. the peaks recurring at the OFDM symbol period nfft+cp
       (autocorrelation of the correlation profile),
    4. the same detection in both halves of the recording.
    Anything less is reported as insufficient evidence.
    """
    n = min(len(x), 1 << 18)
    xx = x[:n]
    p = float((np.abs(xx) ** 2).mean())
    best = {"detected": False, "corr": 0.0, "evidence": {}}
    for nfft in fft_sizes:
        if n < 6 * nfft:
            continue
        c = xx[nfft:] * np.conj(xx[:-nfft])
        for cp in (nfft // 4, nfft // 8):
            if cp < 16:
                continue
            k = np.ones(cp) / cp
            m = np.abs(np.convolve(c, k, mode="valid")) / (p + 1e-12)
            med = float(np.median(m))
            p99 = float(np.percentile(m, 99))
            peaky = (p99 - med) / (med + 1e-9)
            corr = p99
            # peakiness is reported but is NOT a discriminator: continuous
            # OFDM keeps the lag product elevated everywhere (low peaky),
            # while oversampled single carriers can be very peaky - the
            # symbol-period recurrence below separates them
            if corr < 0.45:
                continue
            # evidence 3: peak recurrence at the OFDM symbol period
            mz = m - m.mean()
            period = nfft + cp
            if len(mz) < 3 * period:
                continue
            num = float(np.mean(mz[:-period] * mz[period:]))
            # compare against an off-period lag as the null
            off_lag = period + period // 3
            den = float(np.mean(mz[:-off_lag] * mz[off_lag:]))
            var = float(np.mean(mz * mz)) + 1e-15
            period_score = (num - den) / var
            # evidence 4: both halves agree
            half = len(m) // 2
            c1 = float(np.percentile(m[:half], 99))
            c2 = float(np.percentile(m[half:], 99))
            halves = min(c1, c2) > 0.35
            detected = period_score > 0.3 and halves
            if corr > best["corr"] and detected:
                best = {"detected": True, "fft_size": nfft, "cp_length": cp,
                        "corr": corr,
                        "evidence": {"cp_correlation": round(corr, 3),
                                     "peakiness": round(peaky, 2),
                                     "period_score": round(period_score, 3),
                                     "halves_consistent": bool(halves)}}
            elif corr > best.get("corr", 0) and not best["detected"]:
                best = {"detected": False, "fft_size": nfft, "cp_length": cp,
                        "corr": corr,
                        "evidence": {"cp_correlation": round(corr, 3),
                                     "peakiness": round(peaky, 2),
                                     "period_score": round(period_score, 3),
                                     "halves_consistent": bool(halves),
                                     "verdict": "insufficient evidence"}}
    return best


# --------------------------------------------------------------------------
def estimate_parameters(x: np.ndarray, config, sample_rate=None,
                        rate_ratio: float = 1.0,
                        modulation_hint: str = None,
                        analysis_band_norm: float = None) -> SignalParameters:
    """Run the full S4 battery on a channelised baseband signal.

    Every quantity is published as an :class:`Estimate` - value,
    confidence, method AND validity state - so a consumer can tell a
    measurement from a bound (report §9 rule 1).  ``modulation_hint``,
    when an analyst or a later stage supplies it, lets the moment SNR
    estimator use the right signal kurtosis instead of the
    constant-modulus default.
    """
    p = SignalParameters(sample_rate=sample_rate)

    bw = occupied_bandwidth(x, config.obw_fraction)
    p.obw99_norm = bw["obw99"]
    p.obw3db_norm = bw["obw3db"]
    p.set_estimate("obw", Estimate(
        value=bw["obw99"], confidence=0.9 if bw["obw99"] else 0.0,
        method=bw.get("method", "cumulative PSD (ITU-R SM.443)"),
        state=(EstimateState.VALID if bw["obw99"]
               else EstimateState.UNRESOLVABLE),
        verdict=Verdict.ESTIMATED if bw["obw99"] else Verdict.UNKNOWN))

    cfo = carrier_offset(x)
    p.carrier_offset_norm = cfo["cfo"] if cfo["confident"] else bw["f_center"]
    p.set_estimate("cfo", Estimate(
        value=p.carrier_offset_norm,
        confidence=(min(1.0, cfo["line_snr"] / 20) if cfo["confident"]
                    else 0.3),
        method=(f"x^{cfo['order']} spectral line" if cfo["confident"]
                else "spectral centroid"),
        state=EstimateState.VALID if cfo["confident"] else
        EstimateState.SATURATED,
        detail={"line_snr": round(cfo["line_snr"], 2),
                "order": cfo["order"],
                "reason": ("" if cfo["confident"] else
                           "no usable M-power line: the centroid is a "
                           "fallback, not a carrier measurement")}))

    # independent carrier cross-check (report §10.5 item 3)
    cl = carrier_line(x)
    p.carrier_line_norm = cl.get("carrier_norm")
    p.carrier_line_quality = round(float(cl.get("quality", 0.0)), 1)

    sr = symbol_rate(x, config.symbol_rate_min_norm, obw99=bw["obw99"],
                     obw_guard=getattr(config, "symbol_rate_obw_guard", 0.2))
    if sr["candidates"]:
        p.symbol_rate_norm = sr["candidates"][0]["rate_norm"]
        p.samples_per_symbol = 1.0 / p.symbol_rate_norm
        cands = [c["rate_norm"] for c in sr["candidates"][:8]]
        # bandwidth-implied candidates: a shaped linear modulation's
        # rate sits near OBW99/(1+rolloff) (clean SNR) and near the -3dB
        # bandwidth (noise-robust; OBW99 integrates the noise floor at
        # low SNR). A near-full band suggests a Nyquist-rate recording.
        # At low SNR data lines can outrank the true rate line, so
        # runner-up lines near an implied rate are promoted and the
        # implied rates themselves are appended as hypotheses of last
        # resort for the downstream receiver trials.
        implied = []
        if bw.get("obw99"):
            implied.append(bw["obw99"] / 1.35)
            if bw["obw99"] > 0.45:
                implied.append(0.5)
        if bw.get("obw3db"):
            implied.append(min(bw["obw3db"], 0.5))
        promoted, extras = [], []
        for imp in implied:
            near = [r for r in cands[1:]
                    if abs(r - imp) <= 0.12 * imp and r not in promoted]
            promoted.extend(near)
            if not near and all(abs(imp - e) > 0.05 * imp
                                for e in extras):
                extras.append(imp)
        rest = [r for r in cands[1:] if r not in promoted]
        seen, ordered = set(), []
        for r in cands[:1] + promoted + extras + rest:
            key = round(r, 6)
            if key not in seen:
                seen.add(key)
                ordered.append(r)
        p.symbol_rate_candidates = ordered
        p.set_estimate("symbol_rate", Estimate(
            value=p.symbol_rate_norm, confidence=sr["confidence"],
            method="cyclic periodogram (delay-product sweep)",
            state=EstimateState.VALID,
            detail={"n_candidates": len(ordered),
                    "search_floor_norm": round(sr.get("min_rate", 0.0), 6)}))
    else:
        p.set_estimate("symbol_rate", Estimate(
            value=None, confidence=0.0, method="no cyclic feature found",
            state=EstimateState.UNRESOLVABLE, verdict=Verdict.UNKNOWN,
            detail={"search_floor_norm": round(sr.get("min_rate", 0.0), 6)}))

    fsk = fsk_tones(x)
    if fsk.get("is_fsk"):
        p.fsk_tone_count = fsk["tone_count"]
        p.fsk_deviation_norm = fsk["deviation_norm"]
        p.set_estimate("fsk", Estimate(
            value=float(fsk["tone_count"]),
            confidence=fsk["mass_fraction"],
            method="instantaneous-frequency histogram",
            state=EstimateState.VALID,
            detail={"deviation_norm": fsk["deviation_norm"],
                    "amplitude_cv": round(fsk["amplitude_cv"], 4)}))
        # FSK has a constant envelope, so the |x|^2 cyclic estimator above
        # is unreliable; measure the tone dwell rate instead.
        fr = fsk_symbol_rate(x, config.symbol_rate_min_norm)
        if fr["rate_norm"]:
            p.symbol_rate_norm = fr["rate_norm"]
            p.samples_per_symbol = 1.0 / fr["rate_norm"]
            if fr["rate_norm"] not in p.symbol_rate_candidates:
                p.symbol_rate_candidates.insert(0, fr["rate_norm"])
            p.set_estimate("symbol_rate", Estimate(
                value=fr["rate_norm"], confidence=fr["confidence"],
                method="FSK transition-rate spectral line",
                state=EstimateState.VALID))

    # ---- SNR: two estimators, and the honest state of each ------------
    # The out-of-band noise floor is kurtosis-free and is the primary
    # measurement whenever a noise-only region exists.  The moment
    # estimator is kept as a cross-check and is explicitly marked
    # saturated when it disagrees (report §3).
    #
    # UNITS: ``snr_db`` is Es/N0 - the noise counted in one symbol rate
    # of bandwidth.  That is the quantity the synchronised EVM measures,
    # the quantity that predicts BER, and the only SNR that does not
    # change when the channeliser decimates; the full-band figure the
    # shipped code reported is kept alongside it as
    # ``detail.full_band_snr_db``.  It is computed after the symbol rate
    # because it needs the reference bandwidth.
    # 0.85 of the half-passband keeps the noise measurement inside the
    # channel filter's FLAT region; the transition band is neither
    # signal nor a fair sample of the noise
    spec = spectral_snr(x, reference_bandwidth=p.symbol_rate_norm,
                        band_limit=(0.85 * analysis_band_norm / 2.0
                                    if analysis_band_norm else None))
    ka = signal_kurtosis(modulation_hint) if modulation_hint else 1.0
    ref_full = (spec.detail.get("full_band_snr_db")
                if spec.usable else None)
    mm = m2m4_snr(x, ka, reference_snr_db=ref_full)
    p.estimates["snr_spectral"] = spec.to_dict()
    p.estimates["snr_m2m4"] = mm.to_dict()
    chosen = spec if spec.usable else mm
    if not spec.usable and mm.value is None:
        chosen = Estimate(value=None, confidence=0.0,
                          method="no usable SNR estimator",
                          state=EstimateState.UNRESOLVABLE,
                          verdict=Verdict.UNKNOWN)
    p.snr_db = None if chosen.value is None else round(float(chosen.value), 1)
    p.snr_state = chosen.state.value
    p.snr_method = chosen.method
    p.snr_full_band_db = (None if not spec.usable else
                          spec.detail.get("full_band_snr_db"))
    p.set_estimate("snr", chosen)

    ofdm = ofdm_detect(x)
    if ofdm.get("detected"):
        p.ofdm_detected = True
        p.ofdm_fft_size = ofdm["fft_size"]
        p.ofdm_cp_length = ofdm["cp_length"]
        p.set_estimate("ofdm", Estimate(
            value=float(ofdm["fft_size"]),
            confidence=min(1.0, ofdm["corr"]),
            method="cyclic-prefix autocorrelation",
            state=EstimateState.VALID,
            detail=ofdm.get("evidence", {})))

    # absolute values when the rate is known
    if sample_rate:
        if p.carrier_offset_norm is not None:
            p.carrier_offset_hz = p.carrier_offset_norm * sample_rate
        if p.obw99_norm is not None:
            p.obw99_hz = p.obw99_norm * sample_rate
        if p.symbol_rate_norm is not None:
            p.symbol_rate_hz = p.symbol_rate_norm * sample_rate

    # excess bandwidth from the raised-cosine transition, not from the
    # 99% integral (defect D3)
    # measured on a PSD averaged for THIS purpose: the roll-off fit wants
    # a smooth curve far more than it wants fine resolution
    rf, rp = _rolloff_psd(x, p.symbol_rate_norm)
    if rf is None:
        rf, rp = bw.get("psd_freq"), bw.get("psd_lin")
    # The channelised signal is already centred on its own segment, so
    # the fit is anchored at zero.  Re-centring it on the OBW power
    # centroid instead shifts the model against the data by whatever the
    # noise floor does to that centroid, which measurably biased the
    # result on short records.
    eb = excess_bandwidth(rf, rp, p.symbol_rate_norm, 0.0)
    p.excess_bandwidth = eb["rolloff"]
    p.set_estimate("excess_bandwidth", Estimate(
        value=eb["rolloff"], confidence=0.7 if eb["rolloff"] is not None else 0.0,
        method=eb["method"],
        state=(EstimateState.VALID if eb["rolloff"] is not None
               else EstimateState.UNRESOLVABLE),
        verdict=(Verdict.ESTIMATED if eb["rolloff"] is not None
                 else Verdict.UNKNOWN),
        detail={k: v for k, v in eb.items()
                if k not in ("rolloff", "method")}))

    # snapshot of the front-end answers, so an S6 correction stays visible
    p.symbol_rate_norm_s4 = p.symbol_rate_norm
    p.carrier_offset_norm_s4 = p.carrier_offset_norm
    p.snr_db_s4 = p.snr_db
    return p


def reconcile_with_receiver(p: SignalParameters, demod,
                            tolerance: float = 0.02) -> list:
    """Replace S4 estimates with the values the S6 receiver actually
    recovered, and record every substitution (report §11 item 7).

    The receiver measures the symbol rate and the carrier offset while it
    locks - the GMSK squaring lines and the OQPSK x^2 pair are far more
    accurate than the blind cyclic periodogram - but the shipped pipeline
    kept publishing the S4 figures, which is why WAV-03 reported a 74%
    symbol-rate error for a signal its own demodulator had timed
    correctly.  Returns the list of reconciliation records, which is also
    stored on ``p``.
    """
    notes = []
    locked = bool(getattr(demod, "carrier_locked", False) and
                  getattr(demod, "timing_locked", False))
    status = getattr(demod, "demodulation_status", "FAILED")
    if status == "FAILED":
        return notes

    rs6 = getattr(demod, "symbol_rate_norm_recovered", None)
    if rs6 and getattr(demod, "symbol_rate_confident", False):
        rs4 = p.symbol_rate_norm
        rel = abs(rs6 - rs4) / rs6 if rs4 else float("inf")
        if rel > tolerance:
            notes.append({
                "quantity": "symbol_rate_norm", "s4": rs4, "s6": float(rs6),
                "relative_error": None if rs4 is None else round(rel, 4),
                "method": "receiver timing recovery",
                "reason": ("the receiver locked at a different symbol rate; "
                           "a lock is stronger evidence than a periodogram "
                           "line")})
            p.symbol_rate_norm = float(rs6)
            p.samples_per_symbol = 1.0 / float(rs6)
            if p.sample_rate:
                p.symbol_rate_hz = float(rs6) * p.sample_rate
            p.set_estimate("symbol_rate", Estimate(
                value=float(rs6), confidence=0.95,
                method="recovered by the S6 receiver",
                state=EstimateState.VALID, verdict=Verdict.VALIDATED))

    if getattr(demod, "cfo_confident", False):
        cfo6 = float(getattr(demod, "cfo_applied_norm", 0.0) or 0.0)
        cfo4 = p.carrier_offset_norm
        if cfo4 is None or abs(cfo6 - cfo4) > max(1e-4, 0.05 * abs(cfo6)):
            notes.append({
                "quantity": "carrier_offset_norm", "s4": cfo4, "s6": cfo6,
                "method": "carrier recovered during demodulation",
                "reason": ("the receiver removed a different offset than "
                           "S4 published; the applied value is the one that "
                           "produced the lock")})
            p.carrier_offset_norm = cfo6
            if p.sample_rate:
                p.carrier_offset_hz = cfo6 * p.sample_rate
            p.set_estimate("cfo", Estimate(
                value=cfo6, confidence=0.9,
                method="carrier recovered during demodulation",
                state=EstimateState.VALID, verdict=Verdict.VALIDATED))

    # SNR: the EVM figure only replaces the front-end number when the
    # front-end number is not a real measurement.  Both are Es/N0, so
    # this is a comparison rather than a unit conversion.
    evm = getattr(demod, "evm_percent", None)
    if locked and evm:
        ev = snr_from_evm(evm)
        p.estimates["snr_evm"] = ev.to_dict()
        if ev.usable and (p.snr_state != "valid" or p.snr_db is None):
            notes.append({
                "quantity": "snr_db", "s4": p.snr_db, "s6": ev.value,
                "method": ev.method,
                "reason": (f"the front-end SNR was {p.snr_state}; the "
                           "synchronised EVM is a real measurement")})
            p.snr_db = ev.value
            p.snr_state = "valid"
            p.snr_method = ev.method
            p.set_estimate("snr", ev)

    if notes:
        p.reconciliation.extend(notes)
    return notes
