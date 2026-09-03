"""Stage S2: CFAR-style detection on the time-frequency plane.

Robust per-bin noise floor (median across time), threshold at floor +
margin, binary morphology to join bursts, connected-component labelling
into signal-of-interest boxes. Deterministic, explainable, no training."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..common.models import SignalSegment
from .spectrum import compute_waterfall


def detect_signals(x: np.ndarray, config, sample_rate=None) -> tuple:
    """Returns (list[SignalSegment], debug dict with waterfall etc.)."""
    wf = compute_waterfall(x, nfft=min(config.nfft, 4096), min_rows=16)
    W = wf["waterfall_db"]                       # (time, freq)
    if W.size == 0:
        return [], {"waterfall": wf}
    freqs = wf["freq_norm"]

    # Noise floor: median over time per bin is robust to *bursts*, but a
    # continuous signal raises its bins' medians, so cap the per-bin floor
    # at a global percentile of the band (assumes >~25% of the band is
    # signal-free; strongly coloured noise floors are a documented
    # limitation of this fast path).
    med = np.median(W, axis=0)
    med = ndimage.median_filter(med, size=max(3, config.train_bins * 2 + 1))
    global_floor = float(np.percentile(med, 25))
    floor = np.minimum(med, global_floor + 1.0)
    thresh = floor + config.threshold_db
    mask = W > thresh[None, :]

    # morphological cleanup: close small gaps, drop isolated pixels
    st = np.ones((config.morph_close_size, config.morph_close_size), bool)
    mask = ndimage.binary_closing(mask, structure=st)
    mask = ndimage.binary_opening(mask, structure=np.ones((1, 2), bool))

    labels, n = ndimage.label(mask)
    segments = []
    hop = wf["hop"]
    noise_lin = 10 ** (floor / 10)
    for i, sl in enumerate(ndimage.find_objects(labels)):
        if sl is None:
            continue
        t_sl, f_sl = sl
        n_f = f_sl.stop - f_sl.start
        n_t = t_sl.stop - t_sl.start
        if n_f < config.min_bandwidth_bins or n_t < config.min_duration_frames:
            continue
        block = W[t_sl, f_sl]
        sig_lin = (10 ** (block / 10)).mean()
        nf_lin = noise_lin[f_sl].mean()
        snr = 10 * np.log10(max(sig_lin - nf_lin, 1e-20) / nf_lin)
        if snr < 1.0:
            continue
        start = int(wf["row_start_sample"][t_sl.start])
        stop_row = min(t_sl.stop - 1, len(wf["row_start_sample"]) - 1)
        end = int(wf["row_start_sample"][stop_row] + wf["nfft"])
        segments.append(SignalSegment(
            start_sample=start, end_sample=end,
            f_low_norm=float(freqs[f_sl.start]),
            f_high_norm=float(freqs[min(f_sl.stop, len(freqs) - 1)]),
            snr_db=float(snr),
            confidence=float(min(1.0, snr / 20.0)),
            sample_rate=sample_rate, id=len(segments)))
        if len(segments) >= config.max_signals:
            break
    segments.sort(key=lambda s: -s.snr_db)
    for k, s in enumerate(segments):
        s.id = k
    return segments, {"waterfall": wf, "noise_floor_db": floor,
                      "threshold_db": thresh}
