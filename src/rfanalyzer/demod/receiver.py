"""Stage S6: synchronisation and demodulation to soft bits.

Chain (report S6): coarse CFO removal -> RRC matched filter -> Gardner
timing-recovery loop -> decision-directed carrier PLL -> AGC -> slicer
with max-log LLRs.  FSK takes the instantaneous-frequency path instead.

The auto-driver configures everything from the S4/S5 estimates; every
loop publishes a lock metric so failure is visible, not silent.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

from ..common.models import DemodulationResult
from .constellations import CONSTELLATIONS, slice_symbols
from .filters import rrc_taps


def _coarse_cfo(x: np.ndarray, order: int, limiter: bool = True) -> float:
    """M-power CFO estimate.

    The hard limiter sharpens the spectral line for constant-modulus (PSK)
    signals but destroys the amplitude structure that the x^4 line of QAM
    depends on, so QAM callers pass limiter=False.
    """
    n = min(len(x), 1 << 18)
    base = (x[:n] / (np.abs(x[:n]) + 1e-12)) if limiter else x[:n]
    y = base ** order
    Y = np.abs(np.fft.fft(y * np.hanning(n)))
    Y[0] = 0
    k = int(np.argmax(Y))
    return float(np.fft.fftfreq(n)[k] / order)


def _timing_recover(x: np.ndarray, sps: float,
                    max_symbols: int = 200000) -> tuple:
    """Feedforward Oerder&Meyr timing recovery.

    The squared envelope |x|^2 of a shaped linear modulation contains a
    spectral tone at the symbol rate whose phase encodes the timing offset.
    We measure that phase per block, unwrap it, fit a line (offset + clock
    drift) and interpolate the signal at the fitted symbol instants.
    Fully vectorised and deterministic - no feedback loop to diverge.

    Returns (symbols, tone_strength, locked).
    """
    # bring the stream to an integer number of samples/symbol
    Q = int(round(sps))
    if Q < 4:
        Q = 4
    if abs(sps - Q) / sps > 0.003:
        from fractions import Fraction
        fr = Fraction(Q / sps).limit_denominator(500)
        x = sig.resample_poly(x, fr.numerator, fr.denominator)
    n = len(x)
    if n < 8 * Q:
        return np.zeros(0, dtype=np.complex128), 0.0, False

    env = np.abs(x) ** 2
    block = max(Q * 64, 512)
    n_blocks = max(1, n // block)
    mus, weights, centers = [], [], []
    k = np.exp(-2j * np.pi * np.arange(block) / Q)
    for b in range(n_blocks):
        seg = env[b * block:(b + 1) * block]
        if len(seg) < Q * 8:
            break
        c = (seg * k[:len(seg)]).sum()
        strength = np.abs(c) / (seg.sum() + 1e-12)
        mu = (-np.angle(c) / (2 * np.pi)) * Q   # timing offset in samples
        mus.append(mu)
        weights.append(strength)
        centers.append(b * block)
    mus = np.array(mus)
    weights = np.array(weights)
    tone = float(weights.mean())
    # unwrap modulo-Q jumps between consecutive blocks
    for i in range(1, len(mus)):
        while mus[i] - mus[i - 1] > Q / 2:
            mus[i] -= Q
        while mus[i] - mus[i - 1] < -Q / 2:
            mus[i] += Q
    centers = np.array(centers, dtype=np.float64)
    if len(mus) >= 2:
        # weighted linear fit: mu(t) = mu0 + drift * t  (clock offset)
        A = np.vstack([np.ones_like(centers), centers]).T
        Wd = np.diag(weights / (weights.sum() + 1e-12))
        coef, *_ = np.linalg.lstsq(Wd @ A, Wd @ mus, rcond=None)
        mu0, drift = float(coef[0]), float(coef[1])
    else:
        mu0, drift = float(mus[0]) if len(mus) else 0.0, 0.0
    # symbol sampling instants
    n_sym = min((n - Q) // Q, max_symbols)
    kk = np.arange(n_sym)
    t = mu0 % Q + kk * (Q + drift * Q)
    t = t[(t >= 0) & (t < n - 1)]
    i0 = t.astype(int)
    f = t - i0
    syms = x[i0] * (1 - f) + x[i0 + 1] * f
    locked = bool(tone > 0.02)
    return syms, tone, locked


def _dd_pll(symbols: np.ndarray, modulation: str, loop_bw: float = 0.02) -> tuple:
    """Decision-directed carrier phase/frequency tracking.

    Returns (corrected symbols, phase_error_rms, locked)."""
    table, _ = CONSTELLATIONS[modulation]
    zeta = 0.707
    theta = loop_bw
    d = 1 + 2 * zeta * theta + theta * theta
    alpha = 4 * zeta * theta / d
    beta = 4 * theta * theta / d
    phase = 0.0
    freq = 0.0
    if modulation in ("16QAM", "64QAM", "QPSK"):
        # 4th-power feedforward initial phase (square constellations have
        # E[s^4] at angle pi when axis-aligned); gives the DD loop a warm
        # start so early wrong decisions don't stall convergence
        c4 = (symbols[: 4096] ** 4).mean()
        if np.abs(c4) > 0:
            phase = float((np.angle(c4) - np.pi) / 4.0)
    out = np.empty_like(symbols)
    errs = np.empty(len(symbols))
    for i, s in enumerate(symbols):
        y = s * np.exp(-1j * phase)
        dpt = table[np.argmin(np.abs(y - table))]
        e = float(np.angle(y * np.conj(dpt)))
        freq += beta * e
        freq = float(np.clip(freq, -0.2, 0.2))
        phase += freq + alpha * e
        out[i] = y
        errs[i] = e
    tail = errs[len(errs) // 2:]
    return out, float(np.sqrt((tail ** 2).mean())), bool(np.abs(tail).mean() < 0.35)


def demodulate(x: np.ndarray, modulation: str, sps: float,
               config, cfo_norm: float = None) -> DemodulationResult:
    """Auto-configured demodulation of a channelised baseband signal."""
    x = np.asarray(x, dtype=np.complex128)
    res = DemodulationResult(modulation=modulation, samples_per_symbol=sps)

    if modulation.endswith("FSK"):
        return _demod_fsk(x, modulation, sps, res)
    if modulation not in CONSTELLATIONS:
        res.warnings.append(f"unsupported modulation for demodulation: {modulation}")
        return res

    is_qam = modulation.endswith("QAM")
    order = {"BPSK": 2, "QPSK": 4, "8PSK": 8}.get(modulation, 4)
    # 1. coarse CFO for the *known* modulation. The upstream S4 estimate
    # had to guess the nonlinearity order and can lock a spurious line, so
    # PSK applies it then refines, while QAM (whose x^4 line is reliable
    # only without a limiter) estimates from scratch.
    n = np.arange(len(x))
    cfo = 0.0
    if not is_qam and cfo_norm:
        cfo = float(cfo_norm)
        x = x * np.exp(-2j * np.pi * cfo * n)
    resid = _coarse_cfo(x, order, limiter=not is_qam)
    if abs(resid) < 0.05:
        x = x * np.exp(-2j * np.pi * resid * n)
        cfo += resid
    res.cfo_applied_norm = float(cfo)

    # 2. matched filter
    n_int = max(2, int(round(sps)))
    taps = rrc_taps(n_int, config.rrc_span_symbols, config.rrc_rolloff)
    x = sig.fftconvolve(x, taps, mode="same")

    # keep a short matched-filtered trace for the eye diagram display
    q_eye = max(2, int(round(sps)))
    res.lock_metrics["eye_sps"] = q_eye
    res.eye_trace = np.asarray(x[: 200 * q_eye], dtype=np.complex64)

    # 3. timing recovery (feedforward Oerder&Meyr)
    syms, tone, tlock = _timing_recover(x, sps, config.max_symbols)
    res.timing_locked = tlock
    res.lock_metrics["timing_tone_strength"] = round(tone, 4)
    if len(syms) < 32:
        res.warnings.append("timing recovery produced too few symbols")
        return res

    # 4. AGC
    syms = syms / (np.sqrt((np.abs(syms) ** 2).mean()) + 1e-12)

    # 5. carrier recovery
    syms, perr, clock = _dd_pll(syms, modulation, config.carrier_loop_bw)
    res.carrier_locked = clock
    res.lock_metrics["phase_error_rms"] = round(perr, 4)

    # drop PLL convergence transient
    settle = min(len(syms) // 10, 500)
    syms = syms[settle:]

    # 6. slice
    noise_var = max(1e-4, float(np.var(np.abs(syms)) * 0.5))
    hard, llrs, evm = slice_symbols(syms, modulation, noise_var)
    res.symbols = syms.astype(np.complex64)
    res.hard_bits = hard
    res.llrs = llrs
    res.evm_percent = round(evm, 1)
    if not (tlock and clock):
        res.warnings.append("synchronisation lock is weak; bit stream may be unreliable")
    return res


def _demod_fsk(x: np.ndarray, modulation: str, sps: float,
               res: DemodulationResult) -> DemodulationResult:
    order = int(modulation[0])
    k = int(np.log2(order))
    inst_f = np.diff(np.unwrap(np.angle(x))) / (2 * np.pi)
    # smooth over roughly half a symbol
    w = max(1, int(sps) // 2)
    inst_s = np.convolve(inst_f, np.ones(w) / w, mode="same")
    # tone centres from histogram
    lo, hi = np.percentile(inst_s, [1, 99])
    hist, edges = np.histogram(inst_s, bins=256, range=(lo, hi))
    hist = sig.medfilt(hist.astype(float), 5)
    peaks, _ = sig.find_peaks(hist, height=0.25 * hist.max(), distance=8)
    centers = np.sort(0.5 * (edges[peaks] + edges[peaks + 1]))
    if len(centers) != order:
        res.warnings.append(
            f"expected {order} FSK tones, found {len(centers)}; "
            "using theoretical spacing")
        rs = 1.0 / sps
        centers = (np.arange(order) - (order - 1) / 2) * rs
    # pick the sampling phase that minimises distance to tone centres
    n_int = max(1, int(round(sps)))
    best_phase, best_cost = 0, np.inf
    for p in range(n_int):
        sampled = inst_s[p::n_int]
        d = np.abs(sampled[:, None] - centers[None, :]).min(axis=1)
        c = float(d.mean())
        if c < best_cost:
            best_cost, best_phase = c, p
    sampled = inst_s[best_phase::n_int]
    dists = np.abs(sampled[:, None] - centers[None, :])
    idx = np.argmin(dists, axis=1)
    # tone index -> bits (MSB first)
    bits = np.zeros((len(idx), k), dtype=np.uint8)
    for j in range(k):
        bits[:, j] = (idx >> (k - 1 - j)) & 1
    # LLR per bit from tone-distance margin
    scale = np.diff(centers).mean() if len(centers) > 1 else 1.0
    llrs = np.zeros((len(idx), k), dtype=np.float32)
    for j in range(k):
        mask0 = np.array([(t >> (k - 1 - j)) & 1 == 0 for t in range(order)])
        d0 = dists[:, mask0].min(axis=1)
        d1 = dists[:, ~mask0].min(axis=1)
        llrs[:, j] = ((d1 - d0) / (scale + 1e-12)) * 4
    res.hard_bits = bits.ravel()
    res.llrs = llrs.ravel()
    res.symbols = (sampled / (scale + 1e-12)).astype(np.complex64)
    res.timing_locked = res.carrier_locked = bool(best_cost < 0.25 * scale)
    res.lock_metrics["tone_fit_cost"] = round(float(best_cost), 5)
    res.evm_percent = None
    return res
