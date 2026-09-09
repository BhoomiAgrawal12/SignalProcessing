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
from .constellations import CONSTELLATIONS, MOD_FAMILY, slice_symbols
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


def _spectral_symmetry_cfo(x: np.ndarray, max_samples: int = 1 << 18,
                           search: float = 0.25) -> tuple:
    """Carrier offset from the symmetry of the power spectrum.

    A linear modulation's spectrum is symmetric about its carrier, so the
    carrier is the axis that maximises the correlation between the
    spectrum and its own mirror image.  Nothing in that argument involves
    a moment, which is why it works where the M-power estimators cannot:
    128QAM's E[s^4] is 0.18 and 128APSK's is exactly zero, so their x^4
    spectra carry no carrier line at all and the residual-frequency
    search was ranking data artefacts instead - leaving 128QAM with a 24%
    frequency error and an 8.7% EVM floor at any SNR (report §5.3
    measured the same class of failure on 256QAM).

    Returns (offset in cycles/sample, quality), quality being the peak's
    prominence over the mean correlation.
    """
    n = min(len(x), max_samples)
    if n < 4096:
        return 0.0, 0.0
    nper = min(4096, n // 8)
    f, p = sig.welch(np.asarray(x[:n]), fs=1.0, nperseg=nper,
                     noverlap=nper // 2, return_onesided=False,
                     detrend=False)
    order = np.argsort(f)
    p = np.asarray(p[order], dtype=np.float64)
    nb = len(p)
    p = p - float(np.quantile(p, 0.25))
    p = np.clip(p, 0.0, None)
    if p.sum() <= 0:
        return 0.0, 0.0
    # correlation of the spectrum with its mirror, as a function of the
    # assumed centre: c(k) = sum_j P(k+j) P(k-j), computed for every
    # candidate centre by one FFT-based autocorrelation of the spectrum
    P = np.fft.rfft(p, 2 * nb)
    corr = np.fft.irfft(P * P, 2 * nb)[:nb]      # corr[m] = sum_j p[j]p[m-j]
    # corr index m corresponds to centre bin m/2
    span = int(search * nb)
    centre = nb // 2
    lo = max(0, 2 * (centre - span))
    hi = min(len(corr), 2 * (centre + span) + 1)
    if hi - lo < 8:
        return 0.0, 0.0
    seg = corr[lo:hi]
    k = int(np.argmax(seg))
    mean = float(seg.mean()) + 1e-30
    quality = float(seg[k] / mean)
    # parabolic interpolation for sub-bin resolution
    delta = 0.0
    if 0 < k < len(seg) - 1:
        a, b, c = float(seg[k - 1]), float(seg[k]), float(seg[k + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-30:
            delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    centre_bin = (lo + k + delta) / 2.0
    offset = (centre_bin - centre) / nb
    return float(offset), quality


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
    # Pin the envelope tone frequency before the block-phase fit: the
    # nominal rate can be off by a few percent (near-Nyquist bins, line
    # competition at low SNR), which drifts more than Q/2 per block and
    # aliases the unwrap below. A parabolic peak fit near 1/Q measures
    # the true tone to a fraction of a bin.
    T = float(Q)
    nwin = int(min(n, 1 << 18))
    nfft = int(min(1 << 18, 1 << (int(np.ceil(np.log2(max(nwin, 16)))) + 2)))
    if nwin >= 32 * Q:
        E = np.abs(np.fft.rfft((env[:nwin] - env[:nwin].mean()) *
                               np.hanning(nwin), nfft))
        lo = int(np.floor(0.94 * nfft / Q))
        hi = min(int(np.ceil(1.06 * nfft / Q)) + 1, len(E) - 2)
        if lo >= 2 and hi - lo > 4:
            k0 = lo + int(np.argmax(E[lo:hi]))
            med = float(np.median(E[lo:hi])) + 1e-12
            if E[k0] > 4 * med:
                am, bm, cm = float(E[k0 - 1]), float(E[k0]), float(E[k0 + 1])
                den = am - 2 * bm + cm
                delta = 0.5 * (am - cm) / den if abs(den) > 1e-12 else 0.0
                f_ref = (k0 + float(np.clip(delta, -0.5, 0.5))) / nfft
                if abs(f_ref * Q - 1.0) < 0.08:
                    T = 1.0 / f_ref
    block = max(int(T * 64), 512)
    n_blocks = max(1, n // block)
    cs, weights = [], []
    k = np.exp(-2j * np.pi * np.arange(block) / T)
    for b in range(n_blocks):
        seg = env[b * block:(b + 1) * block]
        if len(seg) < Q * 8:
            break
        c = (seg * k[:len(seg)]).sum()
        # re-reference the block phase to global time zero: the block
        # length is generally NOT a multiple of the (fractional) symbol
        # period, and per-block references would masquerade as a huge
        # fake clock drift of (block mod T)/block per sample
        c = c * np.exp(-2j * np.pi * (b * block) / T)
        weights.append(np.abs(c) / (seg.sum() + 1e-12))
        cs.append(c)
    cs = np.asarray(cs, dtype=np.complex128)
    weights = np.asarray(weights)
    tone = float(weights.mean()) if len(weights) else 0.0
    if len(cs) == 0:
        return np.zeros(0, dtype=np.complex128), 0.0, False
    # complex-domain offset + drift: averaging c_b * conj(c_{b-1}) gives
    # the mean per-block phase increment without an unwrap chain, so one
    # noisy block cannot corrupt every later block's timing estimate
    if len(cs) >= 2:
        # coarse slope from lag-1 products (wrap-safe), then a weighted
        # regression on the detrended residual phases: lag-1 alone has
        # regression-grade bias-freedom but not regression-grade
        # variance, and dense QAM cannot afford the accumulated timing
        # error of a noisy slope
        dphi = float(np.angle((cs[1:] * np.conj(cs[:-1])).sum()))
        idx = np.arange(len(cs), dtype=np.float64)
        det = cs * np.exp(-1j * dphi * idx)
        mean_ang = float(np.angle(det.sum()))
        phi_r = np.angle(det * np.exp(-1j * mean_ang))
        w = weights / (weights.sum() + 1e-12)
        i_m = float((w * idx).sum())
        p_m = float((w * phi_r).sum())
        var = float((w * (idx - i_m) ** 2).sum())
        slope = (float((w * (idx - i_m) * (phi_r - p_m)).sum()) / var
                 if var > 0 else 0.0)
        phi0 = mean_ang + p_m - slope * i_m
        dphi_t = dphi + slope
        drift = -dphi_t / (2 * np.pi) * T / block   # timing samples/sample
        mu0 = float(-phi0 / (2 * np.pi) * T)
    else:
        mu0, drift = float(-np.angle(cs[0]) / (2 * np.pi) * T), 0.0
    # symbol sampling instants, cubic Lagrange interpolation (linear
    # interpolation leaves an ISI floor of several percent EVM, which is
    # irrelevant for QPSK but fatal for 128/256-QAM decisions)
    n_sym = min(int((n - Q) // T), max_symbols)
    kk = np.arange(n_sym)
    t = mu0 % T + kk * (T + drift * T)
    t = t[(t >= 1) & (t < n - 2)]
    i0 = t.astype(int)
    f = t - i0
    xm1, x0, x1, x2 = x[i0 - 1], x[i0], x[i0 + 1], x[i0 + 2]
    c0 = -f * (f - 1) * (f - 2) / 6
    c1 = (f + 1) * (f - 1) * (f - 2) / 2
    c2 = -(f + 1) * f * (f - 2) / 2
    c3 = (f + 1) * f * (f - 1) / 6
    syms = c0 * xm1 + c1 * x0 + c2 * x1 + c3 * x2
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
    # 4th-power feedforward warm start anchored to the constellation's OWN
    # fourth moment: rotate so angle(E[y^4]) matches angle(E[table^4]).
    # This is exact for square QAM, cross QAM and APSK alike (the old
    # assumption of angle pi only held for square constellations).
    ref4 = (table ** 4).mean()
    if np.abs(ref4) > 0.05:
        c4 = (symbols[: 4096] ** 4).mean()
        if np.abs(c4) > 1e-6:
            phase = float((np.angle(c4) - np.angle(ref4)) / 4.0)
    else:
        # constellations with a vanishing 4th moment (APSK rings, dense
        # PSK): probe a grid of rotations inside the constellation's own
        # symmetry sector and start from the best-fitting one
        sym_sector = np.pi / 2
        probe = symbols[: 2048]
        best_phi, best_d = 0.0, np.inf
        for kk in range(16):
            phi = kk * sym_sector / 16
            rot = probe * np.exp(-1j * phi)
            d = np.abs(rot[:, None] - table[None, :]).min(axis=1).mean()
            if d < best_d:
                best_d, best_phi = d, phi
        phase = float(best_phi)
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


def _spectral_centroid_cfo(x: np.ndarray, max_samples: int = 1 << 18,
                           floor_quantile: float = 0.4) -> tuple:
    """Carrier offset of a signal with no usable M-power line.

    Report §5.1: ``_CFO_STRATEGY`` has no entry for FSK, GMSK or analog
    transmissions, and nothing else corrected them, which is why 2FSK,
    4FSK, GMSK and FM all showed large carrier errors.  None of those
    signals produces a clean x^M line - an M-FSK spectrum is a set of
    tones, an FM spectrum is a smeared continuum - but all of them are
    symmetric about their carrier, so the power centroid of the
    above-floor spectrum locates it directly.

    Returns (offset in cycles/sample, quality), where quality is the
    fraction of the band that had to be counted as signal; a compact
    occupancy means the centroid is well determined.
    """
    n = min(len(x), max_samples)
    if n < 1024:
        return 0.0, 0.0
    f, p = sig.welch(np.asarray(x[:n]), fs=1.0,
                     nperseg=min(4096, n // 4), noverlap=None,
                     return_onesided=False, detrend=False)
    order = np.argsort(f)
    f, p = f[order], np.asarray(p[order], dtype=np.float64)
    floor = float(np.quantile(p, floor_quantile))
    excess = np.clip(p - floor, 0.0, None)
    total = float(excess.sum())
    if total <= 0:
        return 0.0, 0.0
    centroid = float((f * excess).sum() / total)
    occupancy = float((excess > 0).mean())
    quality = float(np.clip(1.0 - occupancy, 0.0, 1.0))
    return centroid, quality


# CFO estimation strategy per family: (nonlinearity order, use limiter).
# The hard limiter sharpens PSK lines but destroys amplitude structure, so
# QAM/APSK/ASK run without it. High-order PSK lines (x^16, x^32) are weak
# but usable at the SNRs where those modulations decode at all.
_CFO_STRATEGY = {
    "psk": lambda order: (min(order, 32), True),
    "oqpsk": lambda order: (4, True),
    "qam": lambda order: (4, False),
    "apsk": lambda order: (4, False),
    "ask": lambda order: (2, False),
}

# EVM thresholds (percent) for the quality gate: (good, degraded) per
# modulation; anything worse is FAILED.
#
# CALIBRATION (report §4.2).  The shipped gates were close to the 5G NR
# transmitter limits of 3GPP TS 38.104 Table 6.5.2.2-1, which is a
# defensible starting point but is a TRANSMITTER limit, not a receiver
# decision limit for blind hard slicing.  A Monte-Carlo sweep through
# this project's own slicer (AWGN, perfect synchronisation, 20,000
# symbols) measured the EVM that actually produces a given BER, and
# found four gates dangerously loose: at their `degraded` limits, 32PSK
# reached 7.2% BER, 128QAM 10.8%, 256QAM 10.1% and 128APSK 16.4% - far
# above the ~5% ceiling where blind FEC and interleaver detection still
# works, so the bit layer was being handed unusable bits and spending
# seconds searching them (256QAM at 37 dB: 14.3 s, no frame found).
#
# Each gate below is min(shipped value, measured value).  Taking the
# minimum keeps the conservative 3GPP-derived limits where they were
# already stricter than the BER measurement - a too-strict gate only
# downgrades a good signal to DEGRADED, while a too-loose one admits
# bits that cannot decode - and tightens the four that were unsafe plus
# the three `good` gates (32PSK, 64APSK, 128APSK) the sweep showed
# passing more than 1e-3 BER.
#
# _EVM_GATE_CALIBRATION records the measurement each value came from so
# the next recalibration can see what changed and why.
_EVM_GATE_CALIBRATION = {
    #             shipped      measured EVM at    resulting gate
    #          (good, degr)   (BER 1e-3, 1e-2)   (good, degraded)
    "BPSK":    ((25.0, 45.0), (41.99, 59.18)),
    "QPSK":    ((18.0, 32.0), (31.62, 41.33)),
    "OQPSK":   ((18.0, 32.0), (31.64, 41.51)),
    "8PSK":    ((12.0, 22.0), (17.68, 23.36)),
    "16PSK":   ((7.0, 13.0), (9.36, 12.94)),
    "32PSK":   ((7.5, 12.0), (4.69, 6.80)),
    "OOK":     ((30.0, 50.0), (44.59, 59.33)),
    "4ASK":    ((15.0, 28.0), (19.72, 27.82)),
    "8ASK":    ((10.0, 18.0), (10.00, 13.85)),
    "16QAM":   ((12.0, 20.0), (14.83, 19.34)),
    "32QAM":   ((9.0, 16.0), (9.49, 12.39)),
    "64QAM":   ((7.0, 12.0), (7.05, 9.59)),
    "128QAM":  ((5.0, 9.0), (5.01, 6.19)),
    "256QAM":  ((3.5, 7.0), (3.74, 4.97)),
    "16APSK":  ((10.0, 18.0), (13.31, 17.47)),
    "32APSK":  ((8.0, 14.0), (9.40, 12.38)),
    "64APSK":  ((7.0, 11.0), (5.60, 8.23)),
    "128APSK": ((6.0, 10.0), (4.23, 5.55)),
}

_EVM_GATES = {
    mod: (round(min(shipped[0], measured[0]), 2),
          round(min(shipped[1], measured[1]), 2))
    for mod, (shipped, measured) in _EVM_GATE_CALIBRATION.items()
}


def _order_of(modulation: str) -> int:
    table, k = CONSTELLATIONS[modulation]
    return len(table)


def _apply_status(res: DemodulationResult, modulation: str):
    """Mandatory S6 quality gate: GOOD / DEGRADED / FAILED from EVM and
    lock metrics. A FAILED demodulation must not feed the bit layer."""
    gate = _EVM_GATES.get(modulation)
    if gate is None:                      # fsk/gmsk/analog set it themselves
        return
    evm = res.evm_percent
    locked = res.timing_locked and res.carrier_locked
    if evm is None or not locked:
        res.demodulation_status = "FAILED"
        res.warnings.append("no synchronisation lock: bit stream withheld")
        return
    good, degraded = gate
    if evm <= good:
        res.demodulation_status = "GOOD"
    elif evm <= degraded:
        res.demodulation_status = "DEGRADED"
        res.warnings.append(
            f"EVM {evm}% is marginal for {modulation}: downstream results "
            "are speculative")
    else:
        res.demodulation_status = "FAILED"
        res.warnings.append(
            f"EVM {evm}% is too poor to slice {modulation} reliably: "
            "bit stream withheld from the bit layer")
    res.lock_metrics["evm_gate_good"] = good
    res.lock_metrics["evm_gate_degraded"] = degraded
    # EVM-implied SNR: honest quality figure alongside the M2M4 estimate
    if evm and evm > 0:
        res.lock_metrics["snr_from_evm_db"] = round(
            -20 * np.log10(evm / 100.0), 1)


def _ring_info(table: np.ndarray) -> list:
    """[(radius, n_points)] rings of a constellation (for APSK checks)."""
    radii = np.abs(table)
    rings = []
    for r in sorted(set(np.round(radii, 3))):
        rings.append((float(r), int(np.sum(np.abs(radii - r) < 1e-3))))
    return rings


def _ring_gate_width(table: np.ndarray, radius: float) -> float:
    """Half-width for gating symbols onto one ring: at most 30% of the
    radius but never past the midpoint to the neighbouring ring (dense
    APSK rings sit close and cross-ring leakage biases the phase
    statistic)."""
    radii = sorted(r for r, _n in _ring_info(table))
    gaps = [abs(radius - r) for r in radii if abs(radius - r) > 1e-6]
    min_gap = min(gaps) if gaps else radius
    return float(min(0.3 * radius, 0.45 * min_gap))


def _nearest_distance(syms: np.ndarray, table: np.ndarray,
                      n_probe: int = 2048) -> float:
    """Mean distance to the nearest constellation point (unit-power
    domain). Independent of any spectral-line statistic, so it can
    arbitrate between candidate CFO lines without circularity."""
    z = syms[: n_probe]
    return float(np.abs(z[:, None] - table[None, :]).min(axis=1).mean())


def _rotational_concentration(syms: np.ndarray, modulation: str,
                              family: str) -> float:
    """True carrier-lock evidence: nearest-point EVM is blind to a
    spinning dense-PSK ring, but |E[(s/|s|)^M]| collapses to ~0 when the
    constellation rotates. For APSK the dominant ring's point count sets
    M; for QAM the 4th moment is compared to the table's own."""
    table, _k = CONSTELLATIONS[modulation]
    u = syms / (np.abs(syms) + 1e-12)
    if family in ("psk", "oqpsk"):
        M = len(table) if family == "psk" else 4
        return float(np.abs((u ** M).mean()))
    if family == "apsk":
        rings = _ring_info(table)
        radius, n_pts = max(rings, key=lambda rn: rn[1])
        band = np.abs(np.abs(syms) - radius) < \
            _ring_gate_width(table, radius)
        if band.sum() < 32:
            return 0.0
        return float(np.abs((u[band] ** n_pts).mean()))
    # QAM: a moment-based check is circular with the CFO line selection,
    # so lock evidence comes from geometry: on-grid symbols sit much
    # closer to the table than the same symbols rotated off-grid
    d_meas = _nearest_distance(syms, table)
    d_off = _nearest_distance(syms * np.exp(1j * np.pi / 8), table)
    return float(np.clip((d_off - d_meas) / (d_off + 1e-12), 0.0, 1.0) /
                 0.6)


def _vv_feedforward(syms: np.ndarray, modulation: str, family: str,
                    block: int = 192) -> np.ndarray:
    """Viterbi&Viterbi-style block feedforward carrier recovery.

    Per block, the constellation's rotational statistic (u^M for PSK,
    s^4 against the table reference for QAM/APSK) yields a phase estimate;
    estimates are unwrapped across blocks and interpolated. Being
    feedforward, it cannot cycle-slip - the residual constant M-fold
    ambiguity is exactly what the S7 fan-out enumerates."""
    table, _k = CONSTELLATIONS[modulation]
    order = len(table)
    if family in ("psk", "oqpsk"):
        M = order if family == "psk" else 4
        z = (syms / (np.abs(syms) + 1e-12)) ** M
        ref_angle = float(np.angle((table / np.abs(table)) [0] ** 0)) * 0
        zref = ((table / np.abs(table)) ** M).mean()
        ref_angle = float(np.angle(zref)) if np.abs(zref) > 1e-6 else 0.0
    else:
        M = 4
        z = syms ** 4
        zref = (table ** 4).mean()
        if np.abs(zref) < 0.02:
            # no usable 4th-moment reference (dense APSK): use the
            # dominant ring's point count on ring-gated symbols
            rings = _ring_info(table)
            radius, n_pts = max(rings, key=lambda rn: rn[1])
            M = n_pts
            gate = np.abs(np.abs(syms) - radius) < \
                _ring_gate_width(table, radius)
            u = syms / (np.abs(syms) + 1e-12)
            z = np.where(gate, u ** M, 0)
            tri = table[np.abs(np.abs(table) - radius) < 1e-3]
            zref = ((tri / np.abs(tri)) ** M).mean()
        ref_angle = float(np.angle(zref)) if np.abs(zref) > 1e-6 else 0.0

    n_blocks = max(1, len(z) // block)
    centers, angles = [], []
    for b in range(n_blocks):
        zz = z[b * block:(b + 1) * block]
        m = zz[zz != 0].mean() if np.any(zz != 0) else 0
        if np.abs(m) < 1e-9:
            continue
        centers.append(b * block + block / 2)
        angles.append(np.angle(m) - ref_angle)
    if len(angles) < 1:
        return syms
    ang = np.array(angles, dtype=float)
    # unwrap in the M-fold sector
    for i in range(1, len(ang)):
        while ang[i] - ang[i - 1] > np.pi:
            ang[i] -= 2 * np.pi
        while ang[i] - ang[i - 1] < -np.pi:
            ang[i] += 2 * np.pi
    phi = np.interp(np.arange(len(syms)), centers, ang / M)
    return syms * np.exp(-1j * phi)


def _symbol_domain_cfo(syms: np.ndarray, modulation: str,
                       family: str, extra_candidates: list = None) -> tuple:
    """Residual CFO estimation on symbol-spaced samples.

    After timing recovery the samples are ISI-free, so the M-power
    spectral line is far cleaner than in the oversampled domain - this is
    what makes 16/32-PSK and APSK lockable at all (their sample-domain
    x^16/x^32 lines drown in shaping sidelobes).
    Returns (corrected symbols, applied freq cycles/symbol, line quality).
    """
    table, _k = CONSTELLATIONS[modulation]
    order = len(table)
    if family in ("psk", "oqpsk"):
        Ms = [order if family == "psk" else 4]
    elif family == "apsk":
        Ms = [n for _r, n in _ring_info(table)] + [4]
    elif family == "ask":
        Ms = [2]
    else:
        Ms = [4]
    u = syms / (np.abs(syms) + 1e-12)
    n = len(u)
    nfft = 1 << int(np.ceil(np.log2(max(1024, 4 * n))))
    w = np.hanning(n)
    # Candidate lines can be DATA lines (frame periodicity puts spectral
    # lines into u^M as well), so every candidate is validated by the
    # rotational concentration it produces after correction - the true
    # CFO line freezes the constellation, a data line does not.
    kk = np.arange(n)
    cands = [(0.0, 0.0)]
    for f_extra in (extra_candidates or []):
        if f_extra is not None and abs(f_extra) < 0.45:
            cands.append((float(f_extra), 0.0))
    for M in sorted(set(Ms)):
        if M < 2 or M > 64:
            continue
        base = u if family in ("psk", "oqpsk", "apsk") else syms
        z = (base ** M) * w
        Z = np.abs(np.fft.fft(z, nfft))
        med = float(np.median(Z))
        # the concentration validation below rejects junk candidates, so
        # even modest lines are worth evaluating (dense QAM's s^4 line is
        # real but only a few times the noise median)
        order_idx = np.argsort(Z)[::-1][:10]
        for k in order_idx:
            quality = float(Z[k] / (med + 1e-12))
            if quality < 2.5:
                break
            f = float(np.fft.fftfreq(nfft)[int(k)] / M)
            if abs(f) < 0.45:
                cands.append((f, quality))
    # arbitrate candidates by independent geometric evidence (validating
    # with the same M-power statistic would be circular).
    #
    # The nearest-point score is only a fair test once the scale is
    # right: a few percent of AGC error is enough on a dense grid to
    # rank a wrong frequency above the right one, which is what left
    # 128QAM with a 24% frequency error and an 8.7% EVM floor.  The
    # caller therefore applies the rotation-invariant radius gain
    # (:func:`_radius_gain`) before calling this.
    table_full, _kk2 = CONSTELLATIONS[modulation]
    best_f, best_d, best_q = 0.0, np.inf, 0.0
    for f, q in cands:
        trial = syms[: 2048] * np.exp(-2j * np.pi * f * kk[: 2048]) if f \
            else syms[: 2048]
        # constant rotation must not penalise a correct frequency:
        # align the probe's best constant phase first (cheap grid)
        d = min(_nearest_distance(trial * np.exp(-1j * ph), table_full,
                                  1024)
                for ph in np.linspace(0, np.pi / 2, 12, endpoint=False))
        if d < best_d - 1e-6:
            best_f, best_d, best_q = f, d, q
    if best_f:
        syms = syms * np.exp(-2j * np.pi * best_f * kk)
    return syms, best_f, best_q


def _grid_pitch_gain(symbols: np.ndarray, table: np.ndarray,
                     span: float = 1.25, n_probe: int = 8192) -> tuple:
    """Decision-free scale estimate for grid-structured constellations.

    Every QAM and ASK alphabet places its points on a regular lattice, so
    the projections of the received symbols onto I and Q cluster at odd
    multiples of half the lattice pitch.  The pitch is therefore a
    PERIODICITY, and |E[exp(2*pi*j*f*z)]| peaks sharply at f = 1/pitch
    whatever subset of points the data happens to use - which is exactly
    the property the decision-directed estimate lacks.

    The search span stops short of sqrt(2) on purpose.  A square lattice
    rotated by 45 degrees and scaled by sqrt(2) is a lattice again (its
    own diagonal sub-lattice), and a lattice of pitch d is additionally
    periodic at 2/d because every point then lands on a half-integer.
    Both aliases sit at or beyond sqrt(2), so a span of 1.25 excludes
    them while leaving far more room than the few percent of AGC error
    this is here to remove.

    Returns (gain, concentration); concentration near 1 means the lattice
    is clearly resolved, near 0 means there is no lattice to measure.
    """
    t = np.asarray(table, dtype=np.complex128)
    vals = np.unique(np.round(np.concatenate([t.real, t.imag]), 9))
    if len(vals) < 2:
        return 1.0, 0.0
    pitch = float(np.min(np.diff(vals)))
    if pitch <= 0:
        return 1.0, 0.0
    f0 = 1.0 / pitch
    z = np.asarray(symbols[:n_probe], dtype=np.complex128)
    z = np.concatenate([z.real, z.imag])
    best_f, best_mag = f0, 0.0
    lo, hi, n_grid = f0 / span, f0 * span, 201
    for _stage in range(2):
        fs = np.linspace(lo, hi, n_grid)
        mag = np.abs(np.exp(2j * np.pi * np.outer(fs, z)).mean(axis=1))
        k = int(np.argmax(mag))
        if mag[k] > best_mag:
            best_mag, best_f = float(mag[k]), float(fs[k])
        step = fs[1] - fs[0]
        lo, hi, n_grid = best_f - 2 * step, best_f + 2 * step, 81
    return float(best_f / f0), best_mag


def _radius_gain(symbols: np.ndarray, table: np.ndarray,
                 lo: float = 0.75, hi: float = 1.35,
                 n_probe: int = 2048) -> tuple:
    """Scale estimate from the radius distribution alone.

    Every constellation this receiver supports places its points on a
    small set of radii - two for 16QAM, sixteen for 256QAM, three rings
    for 16APSK - and a symbol's radius is invariant under BOTH a constant
    rotation and a residual frequency error.  Matching the received radii
    to the table's radii therefore measures the AGC's scale error before
    anything about the carrier is known, which is what breaks the
    circularity: the residual-frequency search scores its candidates by
    distance to the constellation, that distance is meaningless at the
    wrong scale, and the scale used to be measurable only after the
    frequency was already right.

    Returns (gain, mean residual radius error after correction).
    """
    t = np.asarray(table, dtype=np.complex128)
    radii = np.unique(np.round(np.abs(t), 9))
    r = np.abs(np.asarray(symbols[:n_probe], dtype=np.complex128))
    if len(r) < 64 or len(radii) < 2:
        return 1.0, 0.0
    # put the probe on the table's own power scale first, so the search
    # window is centred on 1.0
    rms = float(np.sqrt((r ** 2).mean()))
    if rms <= 0:
        return 1.0, 0.0
    r = r / rms * float(np.sqrt((np.abs(t) ** 2).mean()))
    best_g, best_d = 1.0, np.inf
    grid = np.linspace(lo, hi, 121)
    for _stage in range(2):
        for g in grid:
            d = float(np.abs(r[:, None] * g - radii[None, :]).min(axis=1).mean())
            if d < best_d:
                best_d, best_g = d, float(g)
        step = float(grid[1] - grid[0])
        grid = np.linspace(best_g - 2 * step, best_g + 2 * step, 41)
    return best_g, best_d


def _lattice_lock(symbols: np.ndarray, modulation: str,
                  n_phase: int = 32, n_probe: int = 1024) -> tuple:
    """Joint phase and scale acquisition for a lattice constellation.

    Dense QAM cannot acquire phase and scale one at a time: a
    nearest-point phase search run at the wrong scale settles into a
    local minimum, and the lattice-pitch scale estimate has no lattice to
    measure until the constellation is upright.  Both are found together
    by scanning the rotation and keeping the one whose I/Q projections
    are most sharply periodic - the lattice is only periodic when it is
    square to the axes.

    The fourth-moment warm start that serves the other families is not
    enough here: for the cross constellations E[table^4] is weak (0.18
    for 128QAM against 0.68 for 16QAM), so its phase estimate is noisy
    exactly where the scale error hurts most.

    Returns (corrected symbols, phase removed, gain applied).
    """
    table, _k = CONSTELLATIONS[modulation]
    t = np.asarray(table, dtype=np.complex128)
    syms = np.asarray(symbols, dtype=np.complex128)
    if len(syms) < 64:
        return syms, 0.0, 1.0
    probe = syms[:n_probe]
    best = (0.0, 1.0, -1.0)
    for phi in np.linspace(0.0, np.pi / 2, n_phase, endpoint=False):
        g, conc = _grid_pitch_gain(probe * np.exp(-1j * phi), t,
                                   n_probe=n_probe)
        if conc > best[2]:
            best = (float(phi), float(g), float(conc))
    phi, _g0, conc = best
    if conc < 0.2:
        return syms, 0.0, 1.0            # no lattice visible: leave it alone
    syms = syms * np.exp(-1j * phi)
    # refine the rotation inside one coarse step, then measure the pitch
    # once more on the full probe
    step = (np.pi / 2) / n_phase
    fine = np.linspace(-step, step, 33)
    g_ref, _c = _grid_pitch_gain(syms, t)
    best_ph = min(fine, key=lambda ph: _nearest_distance(
        syms[: 2048] * g_ref * np.exp(-1j * ph), t, 2048))
    syms = syms * np.exp(-1j * best_ph)
    phi += float(best_ph)
    gain, _c2 = _grid_pitch_gain(syms, t)
    return syms * gain, float(phi), float(gain)


def _dd_gain(symbols: np.ndarray, modulation: str,
             lo: float = 0.85, hi: float = 1.18,
             coarse: int = 45, passes: int = 4) -> tuple:
    """Decision-directed gain correction.

    The AGC normalises the symbols to unit mean power, which silently
    assumes every constellation point is used equally often.  Real
    traffic never is: a framed stream repeats its sync word, spends most
    of its payload in printable ASCII and pads with zeros, so the mean
    power of the symbols actually transmitted differs from the mean power
    of the table.  Measured on this project's own synthetic frames the
    error is +2.5% for 16QAM and 64QAM, +5.8% for 256QAM and -5.2% for
    128APSK.

    A few percent of scale is harmless for QPSK and fatal for a dense
    constellation: it moves every outer point most of the way to its
    neighbour.  It is the whole of the 5.8% EVM floor that 256QAM showed
    at 60 dB SNR with perfect timing and no carrier offset, and it makes
    the nearest-point EVM read like noise when the symbols are in fact
    within 0.9% of what was sent.  Correcting it is therefore worth more
    to dense constellations than any amount of loop tuning.

    A coarse scan over the scale precedes the least-squares refinement
    because once decisions are wrong the objective has local minima at
    the ratios between constellation rings.

    Returns (rescaled symbols, gain applied).
    """
    table, _k = CONSTELLATIONS[modulation]
    t = np.asarray(table, dtype=np.complex128)
    syms = np.asarray(symbols, dtype=np.complex128)
    if len(syms) < 32:
        return syms, 1.0
    probe = syms[: min(len(syms), 4096)]
    # Coarse stage.  For a lattice constellation (QAM, ASK) the pitch can
    # be measured without decisions at all, which matters because the
    # nearest-point objective is multi-modal on a self-similar grid: a
    # dense square QAM fits itself tolerably at several wrong scales, and
    # a scan over that objective settles into whichever local minimum it
    # started near.  Everything else falls back to the scan.
    fam = MOD_FAMILY.get(modulation)
    g, conc = (1.0, 0.0)
    if fam in ("qam", "ask"):
        g, conc = _grid_pitch_gain(syms, t)
        if not (lo <= g <= hi) or conc < 0.15:
            g, conc = 1.0, 0.0
    unity_d = _nearest_distance(probe, t, len(probe))
    if conc < 0.15:
        grid = np.linspace(lo, hi, coarse)
        g = float(min(grid, key=lambda gg: _nearest_distance(probe * gg, t,
                                                             len(probe))))
        # Never accept a rescale that does not clearly beat leaving the
        # gain alone.  On an unlocked constellation the nearest-point
        # objective is minimised by SHRINKING everything onto the inner
        # points, which looks like a better fit and is in fact a 19%
        # gain error - it is what wrecked 128QAM, whose spinning
        # constellation offers the lattice estimator nothing to measure.
        if _nearest_distance(probe * g, t, len(probe)) > 0.97 * unity_d:
            g = 1.0
    # the search window is deliberately narrow: this corrects the few
    # percent the unit-power AGC gets wrong on non-uniform traffic, not
    # an arbitrary level error, and a wide window only adds ways to fail
    best_d = min(_nearest_distance(probe * g, t, len(probe)), unity_d)
    if best_d == unity_d and g != 1.0:
        g = 1.0
    g = float(np.clip(g, lo, hi))
    for _ in range(max(1, passes)):
        z = probe * g
        dec = t[np.argmin(np.abs(z[:, None] - t[None, :]), axis=1)]
        denom = float((np.abs(probe) ** 2).sum())
        if denom <= 0:
            break
        step = float(np.real((dec * np.conj(probe)).sum()) / denom)
        if not np.isfinite(step) or step <= 0:
            break
        d = _nearest_distance(probe * step, t, len(probe))
        if d >= best_d or not (lo <= step <= hi):
            break
        g, best_d = step, d
    return syms * g, float(g)


def _dd_freq_track(symbols: np.ndarray, modulation: str,
                   passes: int = 2,
                   blocks: tuple = (8, 16, 32, 64, 128)) -> tuple:
    """Remove a residual FREQUENCY error from symbol-spaced samples.

    Report §5.3: for QAM of order >= 128 the receiver ran a static
    96-point phase grid and two phase-polish passes, and nothing in that
    chain can follow a frequency ramp.  A 256QAM capture at 31 dB was
    left with 0.0048 cycles/sample of residual CFO - about 14 degrees per
    symbol at 8 samples/symbol - and sliced at 15% BER while its EVM
    still read a respectable 6.1%, because nearest-point EVM is bounded
    by half the minimum point distance however wrong the decision is.

    The fix the report prescribes: fit a line to the unwrapped
    decision-directed phase error and remove its slope before the normal
    phase polish.  Two details make it work once decisions are already
    partly wrong:

    * the error is averaged over blocks of symbols BEFORE unwrapping, so
      individual wrong decisions cannot create phase jumps;
    * the fit is weighted by each block's decision confidence (how far
      the mean residual sits from a decision boundary), and a pass is
      only kept when it actually reduces the distance to the
      constellation - so a bad fit is discarded instead of applied.

    The block length runs up a ladder.  A long block averages away noise
    but cannot see a fast rotation - it just averages it to zero - while
    a short block sees the rotation and is noisy.  Starting short and
    lengthening pulls in a large residual first and then measures the
    remainder precisely, which is the standard frequency-acquisition
    ladder and the reason this can recover an offset the spectral-line
    estimators missed entirely.

    Returns (corrected symbols, total frequency removed in cycles/symbol).
    """
    table, _k = CONSTELLATIONS[modulation]
    syms = np.asarray(symbols, dtype=np.complex128)
    n = len(syms)
    total = 0.0
    idx = np.arange(n, dtype=np.float64)
    schedule = [b for b in blocks for _ in range(max(1, passes))
                if n >= 8 * b]
    for block in schedule:
        d0 = _nearest_distance(syms, table, min(n, 4096))
        dec = table[np.argmin(np.abs(syms[:, None] - table[None, :]), axis=1)]
        err = syms * np.conj(dec)
        nb = n // block
        if nb < 6:
            continue
        acc = err[: nb * block].reshape(nb, block).sum(axis=1)
        w = np.abs(acc)
        if not np.any(w > 0):
            continue
        ang = np.angle(acc)
        # unwrap the BLOCK-MEAN phase: a per-symbol unwrap chain breaks
        # as soon as one decision is wrong, a block mean does not
        ang = np.unwrap(ang)
        centres = (np.arange(nb) + 0.5) * block
        wn = w / (w.sum() + 1e-30)
        cm = float((wn * centres).sum())
        am = float((wn * ang).sum())
        var = float((wn * (centres - cm) ** 2).sum())
        if var <= 0:
            continue
        slope = float((wn * (centres - cm) * (ang - am)).sum()) / var
        if not np.isfinite(slope) or abs(slope) < 1e-9:
            continue
        trial = syms * np.exp(-1j * slope * idx)
        if _nearest_distance(trial, table, min(n, 4096)) >= 0.98 * d0:
            continue      # not a clear improvement: discard the fit
        syms = trial
        total += slope / (2 * np.pi)   # cycles per symbol
    return syms, float(total)


def _phase_polish(symbols: np.ndarray, modulation: str,
                  window: int = 129) -> np.ndarray:
    """Feedforward fine phase correction: decision-directed error smoothed
    over a sliding window. Removes the residual jitter feedback loops
    leave on dense constellations (128/256-QAM, high-order APSK)."""
    table, _ = CONSTELLATIONS[modulation]
    d2 = np.abs(symbols[:, None] - table[None, :])
    dec = table[np.argmin(d2, axis=1)]
    err = np.angle(symbols * np.conj(dec))
    kernel = np.ones(window)
    # Normalise by the number of taps that actually contributed.  With a
    # fixed 1/window scaling, ``mode="same"`` divides the first and last
    # half-window by the full window even though only part of it overlaps
    # the data, so the correction is attenuated exactly where the
    # estimate is already weakest - and any garbage at the record's edge
    # is then smeared back over half a window of good symbols instead of
    # being averaged away.  Measured on 32PSK: six noise-only symbols at
    # the tail of a 1150-symbol record moved the EVM from 1.1% to 7.6%.
    weight = np.convolve(np.ones_like(err), kernel, mode="same")
    trend = np.convolve(err, kernel, mode="same") / np.maximum(weight, 1.0)
    return symbols * np.exp(-1j * trend)


def demodulate(x: np.ndarray, modulation: str, sps: float,
               config, cfo_norm: float = None) -> DemodulationResult:
    """Auto-configured demodulation, dispatched by modulation family.

    Families: psk / oqpsk / qam / apsk / ask (linear chain with
    family-specific CFO and carrier settings), fsk, gmsk (frequency
    discriminator chain), analog (AM/FM/SSB detectors producing audio,
    not bits). Every path ends in the mandatory quality gate that sets
    res.demodulation_status."""
    x = np.asarray(x, dtype=np.complex128)
    family = MOD_FAMILY.get(modulation)
    if family in ("psk", "oqpsk", "qam", "apsk", "ask") and 0 < sps < 3.0:
        # Oerder&Meyr timing needs about 3+ samples per symbol; near
        # Nyquist-rate recordings are upsampled 2x first (CFO is in
        # cycles/sample, so the normalised offset halves with the rate)
        x = sig.resample_poly(x, 2, 1)
        sps *= 2.0
        if cfo_norm is not None:
            cfo_norm = cfo_norm / 2.0
    res = DemodulationResult(modulation=modulation, samples_per_symbol=sps)
    res.demodulation_status = "FAILED"

    if family == "analog":
        return _demod_analog(x, modulation, res)
    if family == "gmsk":
        return _demod_gmsk(x, sps, res)
    if family == "fsk" or (modulation.endswith("FSK") and
                           modulation[0].isdigit()):
        out = _demod_fsk(x, modulation, sps, res)
        out.demodulation_status = ("GOOD" if out.timing_locked else "DEGRADED")
        if out.hard_bits is None or not len(out.hard_bits):
            out.demodulation_status = "FAILED"
        return out
    if family == "oqpsk":
        return _demod_oqpsk(x, sps, config, res)
    if family not in ("psk", "qam", "apsk", "ask"):
        res.warnings.append(f"unsupported modulation for demodulation: "
                            f"{modulation}")
        return res
    return _demod_linear(x, modulation, family, sps, config, cfo_norm, res)


def _demod_linear(x, modulation, family, sps, config, cfo_norm, res):
    order = _order_of(modulation)
    m_order, limiter = _CFO_STRATEGY[family](order)

    # 1. coarse CFO: upstream estimate for LOW-order constant-modulus
    # families only (the S4 estimator guesses the nonlinearity order and
    # its spurious lines would poison high-order PSK), then a
    # family-appropriate M-power refinement; the precise correction
    # happens later in the symbol domain where the line is clean.
    n = np.arange(len(x))
    cfo = 0.0
    if family in ("psk", "oqpsk") and cfo_norm and order <= 8:
        cfo = float(cfo_norm)
        x = x * np.exp(-2j * np.pi * cfo * n)
    # Sample-domain M-power refinement only where the line is reliable:
    # for dense PSK and APSK the x^M line drowns in shaping sidelobes and
    # a wrong "correction" here poisons the whole chain (timing itself is
    # CFO-invariant, and the symbol-domain estimator after timing is far
    # more sensitive), so those families skip it entirely.
    sample_cfo_reliable = (family in ("psk", "oqpsk") and order <= 8) or \
        family == "ask"
    symmetry_hint = None
    if sample_cfo_reliable:
        resid = _coarse_cfo(x, m_order, limiter=limiter)
        if abs(resid) < 0.05:
            x = x * np.exp(-2j * np.pi * resid * n)
            cfo += resid
        res.lock_metrics["residual_cfo_est"] = round(float(resid), 6)
    else:
        # Moment-free coarse acquisition for the families whose M-power
        # line does not exist: the spectrum is symmetric about the
        # carrier whatever the constellation, so its symmetry axis
        # locates the carrier.  This is the only estimator in the chain
        # that works for 128QAM and 128APSK, whose fourth moments are
        # 0.18 and 0.0 respectively.
        # It is NOT applied blindly: framed traffic puts discrete lines
        # into the spectrum that are not symmetric about the carrier
        # (measured bias: 0.018 cycles/sample on 16QAM, 0.005 on
        # 128QAM), so the value is published as a candidate and a
        # diagnostic, and the symbol-domain stage decides whether it
        # beats the M-power lines on independent geometric evidence.
        sym_f, sym_q = _spectral_symmetry_cfo(x)
        res.lock_metrics["spectral_symmetry_cfo"] = round(float(sym_f), 6)
        res.lock_metrics["spectral_symmetry_quality"] = round(sym_q, 2)
        symmetry_hint = float(sym_f) if sym_q > 1.05 and abs(sym_f) < 0.2 \
            else None
    res.cfo_applied_norm = float(cfo)

    # 2. matched filter
    n_int = max(2, int(round(sps)))
    taps = rrc_taps(n_int, config.rrc_span_symbols, config.rrc_rolloff)
    x = sig.fftconvolve(x, taps, mode="same")

    # eye-diagram trace for the display
    q_eye = max(2, n_int)
    res.lock_metrics["eye_sps"] = q_eye
    res.eye_trace = np.asarray(x[: 200 * q_eye], dtype=np.complex64)

    # 3. timing recovery (feedforward Oerder&Meyr)
    syms, tone, tlock = _timing_recover(x, sps, config.max_symbols)
    res.timing_locked = tlock
    res.lock_metrics["timing_tone_strength"] = round(tone, 4)
    if len(syms) < 32:
        res.warnings.append("timing recovery produced too few symbols")
        return res

    # 4. AGC: coarse power normalisation, then a rotation-invariant scale
    # correction.  The coarse step assumes every constellation point is
    # used equally often and real traffic never provides that - a framed
    # stream repeats its sync word and spends its payload in printable
    # ASCII - so the constellation comes out mis-scaled by a few percent
    # (measured: +2.5% for 16QAM, +5.8% for 256QAM, -5.2% for 128APSK).
    # That is harmless for QPSK and decisive for a dense grid, and it has
    # to be removed HERE because every later stage scores its hypotheses
    # by distance to the constellation.
    syms = syms / (np.sqrt((np.abs(syms) ** 2).mean()) + 1e-12)
    table_r, _ktr = CONSTELLATIONS[modulation]
    g_rad, d_rad = _radius_gain(syms, table_r)
    syms = syms * g_rad
    res.lock_metrics["radius_gain"] = round(float(g_rad), 4)

    # 4b. symbol-domain residual CFO (clean M-power line on ISI-free
    # samples; essential for 16/32-PSK and APSK)
    syms, f_resid, line_q = _symbol_domain_cfo(
        syms, modulation, family,
        extra_candidates=(None if symmetry_hint is None
                          else [symmetry_hint * sps]))
    res.lock_metrics["symbol_cfo_per_symbol"] = round(f_resid, 7)
    res.lock_metrics["symbol_cfo_line_quality"] = round(line_q, 1)

    # 5. carrier recovery.
    # Low-order constellations use the decision-directed PLL (fast, tracks
    # residual frequency). Dense PSK / APSK / dense QAM use slip-free
    # feedforward Viterbi&Viterbi blocks followed by a decision-directed
    # polish; a feedback loop WILL cycle-slip on these and scramble the
    # bit stream even while the EVM still looks fine.
    dense = (family == "psk" and order >= 16) or family == "apsk" or \
        (family in ("qam",) and order >= 128)
    if family == "ask":
        # align the PAM line onto the real axis via the second moment
        c2 = (syms[:4096] ** 2).mean()
        syms = syms * np.exp(-1j * np.angle(c2) / 2)
        syms, perr, clock = _dd_pll(syms, modulation, 0.005)
    elif dense and family == "qam":
        # dense QAM: after the symbol-domain CFO the only unknown is a
        # constant phase. Decision-directed loops and per-block VV both
        # ADD noise here (the s^4 statistic and the decisions are equally
        # unreliable per block), so the phase is found by a fine grid
        # search on nearest-constellation distance and only a very slow
        # feedforward polish tracks residual drift.
        table, _ktab = CONSTELLATIONS[modulation]
        # Phase and scale are entangled on a dense grid: a nearest-point
        # phase search run at the wrong scale finds a local minimum, and
        # a scale search run at the wrong phase has no lattice to
        # measure.  They are broken apart by starting from an estimate
        # that does not depend on the other - the fourth-moment phase,
        # which is completely scale-invariant because a gain g scales
        # E[s^4] by g^4 without touching its argument.
        # Order matters here.  FREQUENCY first: a spinning constellation
        # has no lattice to measure and no constant phase to find, and
        # 128QAM/128APSK have no usable M-power line for the
        # symbol-domain estimator to have removed it (report §5.3).  The
        # decision-directed tracker can do it because the scale is
        # already right from the radius gain, and its short-block rung
        # pulls in up to +-1/16 cycle per symbol.
        syms, f_dd = _dd_freq_track(syms, modulation)
        if f_dd:
            f_resid += f_dd
            res.lock_metrics["dd_freq_track_per_symbol"] = round(f_dd, 7)
            res.lock_metrics["symbol_cfo_per_symbol"] = round(f_resid, 7)
        # then phase and scale together, which is only now well posed
        syms, ph_lat, total_gain = _lattice_lock(syms, modulation)
        res.lock_metrics["lattice_phase_rad"] = round(ph_lat, 5)
        syms, g2 = _dd_gain(syms, modulation)
        total_gain *= g2
        res.lock_metrics["dd_gain_pre_carrier"] = round(total_gain, 4)
        syms = _phase_polish(syms, modulation, 401)
        syms = _phase_polish(syms, modulation, 201)
        d2 = np.abs(syms[:, None] - table[None, :])
        dec = table[np.argmin(d2, axis=1)]
        errs = np.angle(syms * np.conj(dec))
        perr = float(np.sqrt(np.mean(errs[len(errs) // 2:] ** 2)))
        clock = bool(np.abs(errs).mean() < 0.35)
    elif dense:
        table, _ktab = CONSTELLATIONS[modulation]
        # 128APSK's fourth moment is exactly zero and its dominant ring
        # carries the Viterbi&Viterbi anchor, so a residual frequency
        # defeats both; remove it first (report §5.3).
        #
        # NOT for dense PSK: its constellation is a rotationally
        # symmetric ring, so the nearest-point distance that guards this
        # correction is blind to rotation and cannot tell a good fit from
        # a bad one.  Dense PSK has a real x^M line instead, which
        # _symbol_domain_cfo already used.
        f_pre = 0.0
        if family != "psk":
            syms, f_pre = _dd_freq_track(syms, modulation)
        if f_pre:
            f_resid += f_pre
            res.lock_metrics["dd_freq_track_per_symbol"] = round(f_pre, 7)
            res.lock_metrics["symbol_cfo_per_symbol"] = round(f_resid, 7)
        syms = _vv_feedforward(syms, modulation, family)
        if family == "apsk":
            # the VV anchor rides the dominant ring, which fixes phase
            # only modulo that ring's own symmetry (e.g. 30 deg for a
            # 12-point ring) - finer than the full constellation's 90 deg
            # symmetry, so the residual k*(2pi/M_ring) rotation must be
            # resolved against the WHOLE table
            rings = _ring_info(table)
            m_dom = max(rings, key=lambda rn: rn[1])[1]
            probe = syms[: 20000]
            n_sym_full = 2 * np.pi / m_dom
            # the dominant ring is invariant under exactly these rotations
            # and carries NO coset information while dominating the symbol
            # count: gate it OUT and let the other rings discriminate
            radius_dom = max(rings, key=lambda rn: rn[1])[0]
            off_dom = np.abs(np.abs(probe) - radius_dom) > \
                _ring_gate_width(table, radius_dom)
            probe_d = probe[off_dom] if off_dom.sum() > 256 else probe
            cand_ds = []
            # rotations inside the constellation's own symmetry group are
            # identical: only the distinct cosets are candidates
            from .constellations import symmetry_order
            sym = symmetry_order(modulation)
            for kk2 in range(max(1, m_dom // sym)):
                phi = kk2 * n_sym_full
                rot = probe_d * np.exp(-1j * phi)
                d = np.abs(rot[:, None] - table[None, :]).min(axis=1)
                # trimmed mean: outliers (decision-boundary symbols) carry
                # no rotation information and only blur the margin
                d = np.sort(d)[: int(0.8 * len(d))].mean()
                cand_ds.append((float(d), phi))
            cand_ds.sort()
            best_d, best_phi = cand_ds[0]
            margin = (cand_ds[1][0] - best_d) / (best_d + 1e-12)
            res.lock_metrics["apsk_snap_margin"] = round(float(margin), 3)
            if margin < 0.02:
                res.warnings.append(
                    "APSK ring-rotation ambiguity is marginal (snap margin "
                    f"{margin:.3f}); the bit mapping may be rotated")
            if best_phi:
                syms = syms * np.exp(-1j * best_phi)
        syms, f_dd = _dd_freq_track(syms, modulation)
        if f_dd:
            f_resid += f_dd
            res.lock_metrics["dd_freq_track_per_symbol"] = round(f_dd, 7)
            res.lock_metrics["symbol_cfo_per_symbol"] = round(f_resid, 7)
        syms = _phase_polish(syms, modulation, 201)
        syms = _phase_polish(syms, modulation, 65)
        # residual phase error metric from decisions
        d2 = np.abs(syms[:, None] - table[None, :])
        dec = table[np.argmin(d2, axis=1)]
        errs = np.angle(syms * np.conj(dec))
        perr = float(np.sqrt(np.mean(errs[len(errs) // 2:] ** 2)))
        clock = bool(np.abs(errs).mean() < 0.35)
    else:
        loop_bw = config.carrier_loop_bw
        syms, perr, clock = _dd_pll(syms, modulation, loop_bw)
        if order >= 16:
            syms, perr2, clock2 = _dd_pll(syms, modulation, loop_bw / 4)
            perr = min(perr, perr2)
            clock = clock or clock2
        if order >= 64:
            syms = _phase_polish(syms, modulation)
    # rotational concentration: the lock metric nearest-point EVM cannot
    # fake (a spinning ring reads as concentration ~0)
    conc = _rotational_concentration(syms, modulation, family)
    res.lock_metrics["rotational_concentration"] = round(conc, 3)
    if conc < 0.2:
        clock = False
    res.carrier_locked = clock
    res.lock_metrics["phase_error_rms"] = round(perr, 4)

    # 5b. decision-directed gain: run it AFTER carrier recovery, where the
    # decisions it depends on are trustworthy
    syms, gain = _dd_gain(syms, modulation)
    res.lock_metrics["dd_gain"] = round(gain, 4)
    if abs(gain - 1.0) > 0.02:
        res.lock_metrics["dd_gain_note"] = (
            "the transmitted symbols do not use the constellation "
            "uniformly; the unit-power AGC was off by "
            f"{100 * (gain - 1.0):+.1f}%")

    # Discard a guard at BOTH ends.  The head guard lets the loops
    # settle; the tail guard exists because the S2 box has one waterfall
    # row of time resolution, so its end routinely lands a few symbols
    # past the burst, and those few symbols are pure noise sitting where
    # every feedforward estimator has its weakest support.
    settle = min(len(syms) // 10, 500)
    tail = min(len(syms) // 20, 250)
    syms = syms[settle:len(syms) - tail] if tail else syms[settle:]

    # 6. slice to hard bits + LLRs
    noise_var = max(1e-4, float(np.var(np.abs(syms)) * 0.5))
    hard, llrs, evm = slice_symbols(syms, modulation, noise_var)
    res.symbols = syms.astype(np.complex64)
    res.hard_bits = hard
    res.llrs = llrs
    res.evm_percent = round(evm, 1)
    # what the receiver itself recovered, for the S4 reconciliation:
    # the total carrier offset it removed (sample-domain coarse plus the
    # symbol-domain residual referred back to samples) and the rate it
    # actually timed at.
    res.cfo_applied_norm = float(cfo + f_resid / max(sps, 1e-9))
    res.cfo_confident = bool(res.carrier_locked and
                             (sample_cfo_reliable or line_q > 4.0))
    res.symbol_rate_norm_recovered = (1.0 / sps) if sps > 0 else None
    res.symbol_rate_confident = bool(res.timing_locked)
    _apply_status(res, modulation)
    return res


def _demod_oqpsk(x, sps, config, res):
    """OQPSK: the staggered rails mix under any carrier rotation, so the
    carrier must be recovered FIRST (x^4 carries a clean line at 4fc for
    OQPSK because the offset removes the 180-degree transitions), then the
    half-symbol stagger is undone and the signal decodes as QPSK. The
    90-degree ambiguity swaps the rails and thereby flips the stagger
    direction, so both de-offset directions are tried and the one whose
    constellation concentrates wins."""
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x))

    # carrier frequency from the x^4 line
    m = min(len(x), 1 << 16)
    z = (x[:m] ** 4) * np.hanning(m)
    Z = np.abs(np.fft.fft(z))
    k = int(np.argmax(Z))
    quality = float(Z[k] / (np.median(Z) + 1e-12))
    fc = float(np.fft.fftfreq(m)[k] / 4)
    if quality > 15 and abs(fc) < 0.05:
        x = x * np.exp(-2j * np.pi * fc * n)
        res.cfo_applied_norm = float(fc)
    res.lock_metrics["oqpsk_x4_line_quality"] = round(quality, 1)
    # carrier phase from the time-domain 4th moment (pi/2 ambiguous)
    c4 = (x[: 1 << 16] ** 4).mean()
    if np.abs(c4) > 1e-9:
        phi = (np.angle(c4) - np.pi) / 4.0
        x = x * np.exp(-1j * phi)

    # symbol rate: the x^2 spectrum carries a line PAIR at 2fc +- Rs.
    # Data periodicity produces additional strong lines, so only a pair
    # SYMMETRIC about 2fc (known from the x^4 stage; ~0 after the
    # correction above) is accepted.
    z2 = (x[:m] ** 2) * np.hanning(m)
    Z2 = np.abs(np.fft.fft(z2))
    freqs2 = np.fft.fftfreq(m)
    med2 = float(np.median(Z2))
    peak_idx = np.argsort(Z2)[::-1][:12]
    best_pair = None
    for a in range(len(peak_idx)):
        for b in range(a + 1, len(peak_idx)):
            fa, fb = freqs2[peak_idx[a]], freqs2[peak_idx[b]]
            centre = (fa + fb) / 2.0
            rs_c = abs(fa - fb) / 2.0
            if abs(centre) < 0.002 and 1e-3 < rs_c < 0.5:
                q = float(min(Z2[peak_idx[a]], Z2[peak_idx[b]]) / (med2 + 1e-12))
                if q > 10 and (best_pair is None or q > best_pair[1]):
                    best_pair = (rs_c, q)
    if best_pair:
        sps = 1.0 / best_pair[0]
        res.lock_metrics["oqpsk_x2_rs_norm"] = round(best_pair[0], 6)
        res.lock_metrics["oqpsk_x2_pair_quality"] = round(best_pair[1], 1)
    n_int = max(2, int(round(sps)))
    if not (2 <= n_int <= 64):
        n_int = 8
        sps = 8.0

    taps = rrc_taps(n_int, config.rrc_span_symbols, config.rrc_rolloff)
    mf = sig.fftconvolve(x, taps, mode="same")
    half = n_int // 2

    best = None
    for direction in ("advance_q", "advance_i"):
        if direction == "advance_q":
            rail = np.empty_like(mf.imag)
            rail[:-half] = mf.imag[half:]
            rail[-half:] = mf.imag[-1]
            cand = mf.real + 1j * rail
        else:
            rail = np.empty_like(mf.real)
            rail[:-half] = mf.real[half:]
            rail[-half:] = mf.real[-1]
            cand = rail + 1j * mf.imag
        syms, tone, tlock = _timing_recover(cand, sps, config.max_symbols)
        if len(syms) < 32:
            continue
        syms = syms / (np.sqrt((np.abs(syms) ** 2).mean()) + 1e-12)
        syms, f_r, lq = _symbol_domain_cfo(syms, "OQPSK", "oqpsk")
        syms, perr, clock = _dd_pll(syms, "OQPSK", config.carrier_loop_bw)
        conc = float(np.abs(((syms / (np.abs(syms) + 1e-12)) ** 4).mean()))
        if best is None or conc > best["conc"]:
            best = {"syms": syms, "tone": tone, "tlock": tlock,
                    "perr": perr, "clock": clock, "conc": conc,
                    "direction": direction}
    if best is None:
        res.warnings.append("OQPSK timing recovery failed on both rail "
                            "hypotheses")
        return res
    syms = best["syms"]
    settle = min(len(syms) // 10, 500)
    syms = syms[settle:]
    res.timing_locked = best["tlock"]
    res.carrier_locked = best["clock"] and best["conc"] > 0.2
    res.lock_metrics.update({
        "timing_tone_strength": round(best["tone"], 4),
        "phase_error_rms": round(best["perr"], 4),
        "rotational_concentration": round(best["conc"], 3),
        "stagger_direction": best["direction"],
    })
    res.samples_per_symbol = float(sps)
    res.symbol_rate_norm_recovered = 1.0 / sps if sps else None
    res.symbol_rate_confident = bool(
        best_pair is not None and res.timing_locked)
    res.cfo_confident = bool(quality > 15 and res.carrier_locked)
    noise_var = max(1e-4, float(np.var(np.abs(syms)) * 0.5))
    hard, llrs, evm = slice_symbols(syms, "OQPSK", noise_var)
    res.symbols = syms.astype(np.complex64)
    res.hard_bits = hard
    res.llrs = llrs
    res.evm_percent = round(evm, 1)
    _apply_status(res, "OQPSK")
    return res


def _msk_squaring_lines(x: np.ndarray) -> tuple:
    """Classic squaring estimator for h=0.5 CPM (MSK/GMSK): x^2 carries
    two spectral lines at 2fc +- Rs/2, so their spacing IS the symbol
    rate and their midpoint locates the residual carrier.
    Returns (rs_norm, fc_resid_norm, quality)."""
    n = min(len(x), 1 << 17)
    z = (x[:n] ** 2) * np.hanning(n)
    Z = np.abs(np.fft.fftshift(np.fft.fft(z)))
    freqs = np.fft.fftshift(np.fft.fftfreq(n))
    med = float(np.median(Z))
    k1 = int(np.argmax(Z))
    guard = max(4, n // 2048)
    Z2 = Z.copy()
    Z2[max(0, k1 - guard):k1 + guard] = 0
    k2 = int(np.argmax(Z2))
    quality = float(min(Z[k1], Z2[k2]) / (med + 1e-12))
    rs = float(abs(freqs[k1] - freqs[k2]))
    fc = float(0.5 * (freqs[k1] + freqs[k2]) / 2)
    return rs, fc, quality


def _demod_gmsk(x, sps, res):
    """GMSK (h = 0.5): squaring-line symbol rate and carrier, frequency
    discriminator, mid-symbol sign decisions. The discriminator sign IS
    the bit for MSK-family signals, so no carrier loop is needed."""
    # coarse carrier first: the squaring-line pair is only resolvable
    # once the residual offset is small enough that both lines stay
    # inside the analysis band (report §5.1)
    coarse, cq = _spectral_centroid_cfo(x)
    if abs(coarse) > 1e-4 and cq > 0.3:
        x = x * np.exp(-2j * np.pi * coarse * np.arange(len(x)))
        res.lock_metrics["coarse_cfo_centroid"] = round(float(coarse), 6)
    else:
        coarse = 0.0
    # Independent symbol-clock evidence, from the receiver's own data.
    # The x^2 line PAIR exists for anything that puts energy either side
    # of a carrier - an analog FM signal at a deviation index near 0.5
    # produces a perfectly good pair AND a discriminator margin near the
    # 1.0 that says "h = 0.5".  What it cannot produce is a SYMBOL CLOCK.
    # The tone-transition process |d(inst)/dt| carries a spectral line at
    # the symbol rate for any keyed signal and nothing for a continuously
    # modulated one, so comparing the two rates separates a digital CPM
    # signal from an analog carrier without relying on S4 having been
    # right about the rate.
    from ..params.estimators import fsk_symbol_rate
    tr = fsk_symbol_rate(x)
    rs_tr = tr.get("rate_norm")
    res.lock_metrics["transition_rate_norm"] = (
        None if rs_tr is None else round(float(rs_tr), 6))
    res.lock_metrics["transition_rate_confidence"] = round(
        float(tr.get("confidence", 0.0)), 3)

    rs, fc, quality = _msk_squaring_lines(x)
    if quality > 20 and 1e-3 < rs < 0.5:
        n_int = int(round(1.0 / rs))
        x = x * np.exp(-2j * np.pi * fc * np.arange(len(x)))
        res.lock_metrics["squaring_rs_norm"] = round(rs, 6)
        res.lock_metrics["squaring_line_quality"] = round(quality, 1)
        if rs_tr:
            # h=0.5 keying makes these two measurements the same number
            res.lock_metrics["squaring_vs_transition"] = round(
                float(rs / rs_tr), 3)
        res.symbol_rate_norm_recovered = float(rs)
        res.symbol_rate_confident = True
        res.cfo_applied_norm = float(coarse + fc)
        res.cfo_confident = True
    else:
        n_int = int(round(sps))
        fc = 0.0
        res.cfo_applied_norm = float(coarse)
        res.cfo_confident = bool(coarse and cq > 0.5)
        res.warnings.append("GMSK squaring lines not found: falling back "
                            "to the upstream symbol-rate estimate")
    if not (2 <= n_int <= max(64, int(round(sps)) + 1)):
        n_int = 8

    inst = np.diff(np.unwrap(np.angle(x))) / (2 * np.pi)
    inst = inst - np.median(inst)

    def sample_at(n_i):
        w = max(2, n_i // 2)
        sm = np.convolve(inst, np.ones(w) / w, mode="same")
        best_p, best_score = 0, -1.0
        for p2 in range(n_i):
            sc = float(np.abs(sm[p2::n_i]).mean())
            if sc > best_score:
                best_score, best_p = sc, p2
        return sm[best_p::n_i], best_score / (0.25 / n_i)

    samp, margin0 = sample_at(n_int)
    dev = 0.25 / n_int                     # theoretical peak deviation
    res.samples_per_symbol = float(n_int)
    res.hard_bits = (samp > 0).astype(np.uint8)
    res.llrs = (-samp / (dev + 1e-12) * 4).astype(np.float32)
    res.symbols = (samp / (dev + 1e-12)).astype(np.complex64)
    margin = float(np.abs(samp).mean() / (dev + 1e-12))
    res.timing_locked = res.carrier_locked = bool(margin > 0.5)
    res.lock_metrics["discriminator_margin"] = round(margin, 3)
    # Binary-level check.  MSK-family signalling is BINARY: the
    # discriminator output has one magnitude and two signs, so |samp|
    # is unimodal.  An M-ary FSK signal has M/2 distinct magnitudes and
    # spreads |samp| out.  Without this the GMSK hypothesis scores well
    # on a 4FSK capture - the discriminator swings hard either way - and
    # a strong discriminator margin alone cannot tell them apart.
    mags = np.abs(samp)
    res.lock_metrics["level_spread"] = round(
        float(mags.std() / (mags.mean() + 1e-12)), 4)
    res.demodulation_status = ("GOOD" if margin > 0.7 else
                               "DEGRADED" if margin > 0.4 else "FAILED")
    if res.demodulation_status == "FAILED":
        res.warnings.append("GMSK discriminator margin too low: bit stream "
                            "withheld")
    return res


def _demod_analog(x, modulation, res):
    """Analog demodulators: audio out, no bit stream. The bit layer is
    skipped for these signals by design."""
    n = len(x)
    audio = None
    extra = {}
    if modulation == "FM":
        # centre the carrier before the discriminator: an uncorrected
        # offset appears as a DC term on the audio and biases the
        # deviation measurement (report §5.1)
        coarse, cq = _spectral_centroid_cfo(x)
        if abs(coarse) > 1e-4 and cq > 0.2:
            x = x * np.exp(-2j * np.pi * coarse * np.arange(n))
            res.cfo_applied_norm = float(coarse)
            res.cfo_confident = True
            extra["coarse_cfo_centroid"] = round(float(coarse), 6)
        inst = np.diff(np.unwrap(np.angle(x))) / (2 * np.pi)
        residual = float(np.median(inst))
        audio = inst - residual
        res.cfo_applied_norm = float(res.cfo_applied_norm + residual)
        extra["discriminator_dc_norm"] = round(residual, 6)
        extra["peak_deviation_norm"] = round(float(np.percentile(
            np.abs(audio), 99)), 5)
    elif modulation in ("AM-DSB-WC", "AM-DSB-SC"):
        if modulation == "AM-DSB-WC":
            # the carrier IS a spectral line: find it and centre on it,
            # otherwise the envelope detector sees a beat note
            X = np.abs(np.fft.fft(x[:min(n, 1 << 17)]))
            kk = int(np.argmax(X))
            f0 = float(np.fft.fftfreq(len(X))[kk])
            if abs(f0) > 1e-5:
                x = x * np.exp(-2j * np.pi * f0 * np.arange(n))
            res.cfo_applied_norm = f0
            res.cfo_confident = True
            extra["carrier_line_norm"] = round(f0, 6)
            env = np.abs(x)
            audio = env - env.mean()
            carrier = float(env.mean() / (env.std() + 1e-12))
            extra["carrier_to_modulation"] = round(carrier, 2)
        else:
            # suppressed carrier: recover the carrier from the squared
            # spectrum line, then take the coherent (real) component
            cfo = _coarse_cfo(x, 2, limiter=False)
            y = x * np.exp(-2j * np.pi * cfo * np.arange(n))
            rot = np.angle((y[:65536] ** 2).mean()) / 2
            audio = (y * np.exp(-1j * rot)).real
            res.cfo_applied_norm = float(cfo)
            res.cfo_confident = True
            extra["recovered_carrier_norm"] = round(float(cfo), 6)
    else:                                   # SSB
        # centre the occupied band; absolute audio pitch is ambiguous for
        # suppressed-carrier SSB and reported as such
        X = np.abs(np.fft.fft(x[:min(n, 1 << 17)]))
        k = int(np.argmax(X))
        f0 = np.fft.fftfreq(len(X))[k]
        y = x * np.exp(-2j * np.pi * f0 * np.arange(n))
        audio = y.real
        res.cfo_applied_norm = float(f0)
        res.cfo_confident = bool(not modulation.endswith("SC"))
        extra["shifted_by_norm"] = round(float(f0), 6)
        if modulation.endswith("SC"):
            res.warnings.append("suppressed-carrier SSB: absolute audio "
                                "pitch is ambiguous without the carrier")
    # post-detection audio low-pass: the discriminator/envelope output
    # carries wideband noise (FM click noise especially) far above the
    # message band; audio lives well below ~0.05 cycles/sample here
    lpf = sig.firwin(129, 0.05)
    audio = sig.fftconvolve(audio, lpf, mode="same")
    audio = audio / (np.abs(audio).max() + 1e-12)
    # audio quality: tonal concentration of the demodulated spectrum
    A = np.abs(np.fft.rfft(audio[:min(len(audio), 1 << 16)]))
    conc = float(np.sort(A)[-8:].sum() / (A.sum() + 1e-12))
    res.lock_metrics.update({"audio_tonal_concentration": round(conc, 3),
                             **extra})
    res.audio = audio.astype(np.float32)
    res.demodulation_status = ("GOOD" if conc > 0.15 else
                               "DEGRADED" if conc > 0.05 else "FAILED")
    res.timing_locked = res.carrier_locked = res.demodulation_status == "GOOD"
    res.warnings.append("analog transmission: audio demodulated, no bit "
                        "layer applies")
    return res


def _demod_fsk(x: np.ndarray, modulation: str, sps: float,
               res: DemodulationResult) -> DemodulationResult:
    order = int(modulation[0])
    k = int(np.log2(order))
    # coarse carrier: an M-FSK spectrum is symmetric about its carrier,
    # so the power centroid removes the bulk of the offset; the tone
    # centres measured below then refine it (report §5.1 - the shipped
    # chain had no CFO path at all for FSK, so the reported offset was
    # always zero however far the signal actually sat)
    coarse, cq = _spectral_centroid_cfo(x)
    if abs(coarse) > 1e-4 and cq > 0.3:
        x = x * np.exp(-2j * np.pi * coarse * np.arange(len(x)))
        res.lock_metrics["coarse_cfo_centroid"] = round(float(coarse), 6)
    else:
        coarse = 0.0
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
    # tone-fit quality in units of the tone spacing: 0 is a perfect fit,
    # 0.5 is a symbol landing exactly between two tones.  This is the
    # FSK analogue of EVM and is what the S5 receiver trial scores.
    res.lock_metrics["tone_fit_ratio"] = round(
        float(best_cost / (scale + 1e-12)), 4)
    # residual carrier: the midpoint of the recovered tone set
    residual = float(np.mean(centers))
    res.cfo_applied_norm = float(coarse + residual)
    res.cfo_confident = bool(res.timing_locked)
    res.lock_metrics["tone_centre_residual"] = round(residual, 6)
    res.symbol_rate_norm_recovered = (1.0 / n_int) if n_int else None
    res.symbol_rate_confident = bool(res.timing_locked)
    res.evm_percent = None
    return res
