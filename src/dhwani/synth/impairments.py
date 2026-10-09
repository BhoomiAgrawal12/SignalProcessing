"""Channel impairments with known ground truth."""
from __future__ import annotations

import numpy as np
from scipy import signal as sig


def awgn(x: np.ndarray, snr_db: float, rng) -> np.ndarray:
    p = (np.abs(x) ** 2).mean()
    nvar = p / (10 ** (snr_db / 10))
    n = rng.normal(0, np.sqrt(nvar / 2), (len(x), 2))
    return x + (n[:, 0] + 1j * n[:, 1])


def freq_offset(x: np.ndarray, cfo_norm: float, phase: float = 0.0) -> np.ndarray:
    n = np.arange(len(x))
    return x * np.exp(1j * (2 * np.pi * cfo_norm * n + phase))


def phase_noise(x: np.ndarray, std_rad_per_sample: float, rng) -> np.ndarray:
    walk = np.cumsum(rng.normal(0, std_rad_per_sample, len(x)))
    return x * np.exp(1j * walk)


def timing_offset(x: np.ndarray, frac: float) -> np.ndarray:
    """Fractional-sample delay via FFT phase ramp."""
    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N)
    return np.fft.ifft(X * np.exp(-2j * np.pi * f * frac))


def clock_offset(x: np.ndarray, ppm: float) -> np.ndarray:
    """Sampling-clock offset: resample by (1 + ppm*1e-6)."""
    ratio = 1.0 + ppm * 1e-6
    n_out = int(len(x) / ratio)
    t = np.arange(n_out) * ratio
    i0 = np.floor(t).astype(int)
    frac = t - i0
    i0 = np.clip(i0, 0, len(x) - 2)
    return x[i0] * (1 - frac) + x[i0 + 1] * frac


def iq_imbalance(x: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
    g = 10 ** (gain_db / 20)
    ph = np.radians(phase_deg)
    i = x.real
    q = g * (x.imag * np.cos(ph) + x.real * np.sin(ph))
    return i + 1j * q


def dc_offset(x: np.ndarray, level: complex) -> np.ndarray:
    return x + level


def multipath(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    return sig.lfilter(taps, [1.0], x)


def cfo_drift(x: np.ndarray, drift_norm: float) -> np.ndarray:
    """Carrier drifting linearly by drift_norm cycles/sample over the burst
    (oscillator warm-up, Doppler): phase = pi * drift/N * n^2."""
    n = np.arange(len(x))
    return x * np.exp(1j * np.pi * drift_norm / max(1, len(x)) * n ** 2)


def cw_interferer(x: np.ndarray, freq_norm: float, rel_db: float) -> np.ndarray:
    """An unmodulated carrier at freq_norm, rel_db relative to the signal."""
    p = float((np.abs(x) ** 2).mean()) * 10 ** (rel_db / 10)
    return x + np.sqrt(p) * np.exp(2j * np.pi * freq_norm * np.arange(len(x)))
