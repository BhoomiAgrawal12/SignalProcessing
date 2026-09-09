"""S1-S3: conditioning, multi-resolution detection, channelisation.

The front end had no direct test coverage at all (report §6), and it is
where every measured failure in §10 lives.
"""
import numpy as np
import pytest
from scipy import signal as sg

from rfanalyzer.channelization import channelize
from rfanalyzer.common.config import Config
from rfanalyzer.conditioning import condition
from rfanalyzer.demod.filters import rrc_taps
from rfanalyzer.detection import detect_signals
from rfanalyzer.detection.cfar import (_merge_close_bands, _prune_skirts,
                                       _reference_smooth_continuum)


def _burst(n_sym, sps, rng, beta=0.35):
    syms = (rng.integers(0, 2, n_sym) * 2 - 1) + \
        1j * (rng.integers(0, 2, n_sym) * 2 - 1)
    up = np.zeros(n_sym * sps, dtype=np.complex128)
    up[::sps] = syms
    x = sg.fftconvolve(up, rrc_taps(sps, 10, beta), mode="same")
    return x / np.sqrt((np.abs(x) ** 2).mean())


# ------------------------------------------------------- S1 conditioning
def test_real_input_is_converted_to_the_analytic_signal():
    """A mono WAV is real, so its spectrum is mirrored about DC.  Nothing
    used to remove that mirror, which is why the AO-73 detector returned
    the whole 16.4 kHz audio band as one signal (report §10.2)."""
    n = 40000
    x = np.cos(2 * np.pi * 0.1 * np.arange(n)).astype(np.complex64)
    y, rep = condition(x)
    assert rep.analytic_conversion
    X = np.abs(np.fft.fft(np.asarray(y[:8192])))
    positive, negative = X[:4096].max(), X[4096:].max()
    assert negative < 0.01 * positive          # mirror suppressed


def test_complex_input_is_not_touched_by_the_analytic_path(rng):
    x = (rng.normal(0, 1, 20000) + 1j * rng.normal(0, 1, 20000)
         ).astype(np.complex64)
    _y, rep = condition(x)
    assert not rep.analytic_conversion


# ---------------------------------------------------------- S2 detection
def test_narrowband_signal_inside_a_wide_band_is_isolated(rng):
    """The §10 capability gap, in miniature: a signal occupying 4% of the
    band must come back as its own segment, not as the whole band."""
    n = 1 << 17
    sps = 40
    sig = _burst(n // sps, sps, rng)[:n]
    sig = sig * np.exp(2j * np.pi * 0.12 * np.arange(len(sig)))
    x = sig + (rng.normal(0, 0.05, (len(sig), 2)) @ np.array([1, 1j]))
    segs, dbg = detect_signals(np.asarray(x, dtype=np.complex64),
                               Config().cfar)
    assert segs
    best = segs[0]
    assert best.bandwidth_norm < 0.25            # not the whole band
    assert best.f_low_norm <= 0.12 <= best.f_high_norm
    assert dbg["chosen_resolution"] > 0
    assert dbg["chosen_noise_model"]


def test_detection_tracks_a_sloped_noise_floor(rng):
    """An audio-band capture has a strongly coloured floor; a flat
    threshold puts most of the band above it."""
    n = 1 << 17
    noise = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    # give the noise a 30 dB tilt across the band
    N = np.fft.fft(noise)
    tilt = np.linspace(1.0, 0.03, len(N))
    noise = np.fft.ifft(N * np.fft.fftshift(tilt))
    sps = 16
    sig = _burst(n // sps, sps, rng)[:n] * 3.0
    sig = sig * np.exp(2j * np.pi * -0.2 * np.arange(len(sig)))
    segs, _ = detect_signals(np.asarray(sig + noise, dtype=np.complex64),
                             Config().cfar)
    assert segs
    assert segs[0].bandwidth_norm < 0.4
    assert segs[0].f_low_norm <= -0.2 <= segs[0].f_high_norm


def test_segments_come_back_ranked(rng):
    n = 1 << 16
    x = (rng.normal(0, 0.05, (n, 2)) @ np.array([1, 1j]))
    for fc, amp in ((0.2, 4.0), (-0.3, 1.5)):
        s = _burst(n // 16, 16, rng)[:n] * amp
        x = x + s * np.exp(2j * np.pi * fc * np.arange(n))
    segs, _ = detect_signals(np.asarray(x, dtype=np.complex64), Config().cfar)
    assert len(segs) >= 2
    scores = [s.rank_score for s in segs]
    assert scores == sorted(scores, reverse=True)
    assert [s.id for s in segs] == list(range(len(segs)))


def test_a_clean_signal_does_not_shatter_into_skirts(rng):
    """Defect D5: a clean burst produced a second small CFAR segment in
    its own shaping skirt."""
    n = 1 << 16
    sig = _burst(n // 8, 8, rng)[:n]
    x = sig + (rng.normal(0, 0.02, (len(sig), 2)) @ np.array([1, 1j]))
    segs, _ = detect_signals(np.asarray(x, dtype=np.complex64), Config().cfar)
    assert segs
    assert segs[0].power_fraction > 0.8
    strong = [s for s in segs if s.power_fraction > 0.05]
    assert len(strong) == 1


def test_merge_and_prune_helpers():
    bands = [
        {"f_low": 0.0, "f_high": 0.1, "bandwidth": 0.1, "snr_db": 20.0,
         "power_fraction": 0.9, "density_gain_db": 9.5,
         "concentration_db": 8.6, "nper": 1024},
        {"f_low": 0.101, "f_high": 0.11, "bandwidth": 0.009, "snr_db": 5.0,
         "power_fraction": 0.005, "density_gain_db": 0.0,
         "concentration_db": -20.0, "nper": 1024},
    ]
    merged = _merge_close_bands([dict(b) for b in bands], 0.5)
    assert len(merged) == 1                    # the sliver is adjacent
    kept, dropped = _prune_skirts([dict(b) for b in bands])
    assert len(kept) == 1 and len(dropped) == 1


def test_continuum_reference_fits_under_a_wide_emission(rng):
    """A running quantile is lifted by an emission wider than its own
    window; the continuum fit rejects it as an outlier instead.

    The noise scatter is part of the operating condition, not decoration:
    the sigma clipping needs a residual scale to measure outliers
    against, and a perfectly smooth synthetic floor gives it none."""
    n = 4096
    floor = np.linspace(-100.0, -90.0, n)
    lvl = floor + rng.normal(0.0, 1.5, n)       # chi-square-ish scatter
    lvl[1400:2600] += 25.0                      # 30% of the band is signal
    ref = _reference_smooth_continuum(lvl)
    # tracks the sloped floor away from the emission ...
    assert np.max(np.abs(ref[:1200] - floor[:1200])) < 4.0
    # ... and stays underneath it, not on top of it
    assert np.all(ref[1800:2200] < floor[1800:2200] + 8.0)


# ------------------------------------------------------ S3 channelisation
def test_channel_filter_rejects_a_neighbouring_emission(rng):
    """Without a real channel filter the only band limiting is the
    decimator's anti-alias filter, so a neighbour survives into the
    "channelised" signal and every S4 estimate measures the wrong thing
    (report §10.2)."""
    n = 1 << 16
    sps = 16
    wanted = _burst(n // sps, sps, rng)[:n] * np.exp(
        2j * np.pi * -0.15 * np.arange(n))
    other = _burst(n // sps, sps, rng)[:n] * np.exp(
        2j * np.pi * 0.15 * np.arange(n))
    x = np.asarray(wanted + other, dtype=np.complex64)
    segs, _ = detect_signals(x, Config().cfar)
    seg = next(s for s in segs if s.f_low_norm <= -0.15 <= s.f_high_norm)
    ch = channelize(x, seg)
    assert ch["channel_filter_applied"]
    y = ch["samples"]
    f, p = sg.welch(y, nperseg=2048, return_onesided=False, detrend=False)
    keep = p[np.abs(f) < 0.15].sum()
    reject = p[np.abs(f) > 0.35].sum()
    assert reject < 0.02 * keep


def test_channelizer_reports_its_analysis_band(rng):
    n = 1 << 15
    x = np.asarray(_burst(n // 8, 8, rng)[:n], dtype=np.complex64)
    segs, _ = detect_signals(x, Config().cfar)
    ch = channelize(x, segs[0])
    assert 0 < ch["analysis_band_norm"] <= 1.0


def test_burst_trim_excludes_the_noise_tail(rng):
    """A handful of noise samples at the burst edge cost dense PSK 6.5%
    EVM, so the trim moves inward rather than outward.

    The check is on the CONTENT of the channelised output: its first and
    last samples must be at burst level, not at noise level."""
    n_sig = 1 << 14
    sig = _burst(n_sig // 8, 8, rng)[:n_sig]
    pad = rng.normal(0, 1e-3, (3000, 2)) @ np.array([1, 1j])
    x = np.asarray(np.concatenate([pad, sig, pad]), dtype=np.complex64)
    segs, _ = detect_signals(x, Config().cfar)
    ch = channelize(x, segs[0])
    y = np.abs(np.asarray(ch["samples"]))
    level = float(np.median(y))
    edge = float(np.concatenate([y[:64], y[-64:]]).mean())
    assert edge > 0.2 * level          # edges carry signal, not pad
    assert len(ch["samples"]) < len(x)  # something really was trimmed
