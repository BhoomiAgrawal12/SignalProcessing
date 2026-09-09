"""Stage S2: multi-resolution signal detection on the time-frequency plane.

The shipped detector thresholded a single STFT against a per-bin median
that was clipped to a global percentile of the band.  That works on a
synthetic capture with a flat noise floor and fails on a real one: on
``SelfRun/ao73.wav`` - a 48 kHz SSB audio recording holding a 2 kHz,
1200 baud BPSK carrier at 1102 Hz - it returned ONE segment spanning the
entire 16.4 kHz audio band, and no downstream search could recover what
that segmentation discarded (report §10.2).

Three things are different here:

1. **A colour-adaptive noise reference.**  The floor is a running low
   quantile over frequency (an order-statistic CFAR reference), so a
   sloped or humped noise floor - the normal case for audio-band SSB, or
   for any receiver with a shaped IF - no longer puts the whole band
   above threshold.
2. **Several analysis resolutions.**  A narrowband carrier inside a much
   wider capture is invisible at a coarse resolution and shatters into
   noise slivers at too fine a one.  Every resolution produces a complete
   segmentation and the one with the most compact, highest-contrast box
   wins, scored by *spectral density gain*: how much denser the box is
   than the average of the band it sits in.  The whole band scores 0 dB
   by construction, so "everything is the signal" can never win against a
   real detection.
3. **An independent carrier cross-check.**  The x^2 line locates a
   suppressed-carrier BPSK directly.  When it disagrees with every
   detected box, a candidate box is synthesised around it and the
   disagreement is recorded rather than ignored (report §10.5 item 3).

Segments come back as a RANKED list (report §9 layer A); the pipeline is
free to try the runners-up instead of committing to ``segments[0]``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal as sig

from ..common.models import SignalSegment
from .spectrum import compute_waterfall


# Analysis resolutions, in Welch segment lengths.  The ladder spans four
# octaves: 256 bins resolve a wideband burst in a short capture, 32768
# bins resolve a 40 Hz carrier inside a 48 kHz recording.
_RESOLUTION_LADDER = (256, 1024, 4096, 16384, 32768)


def _welch_psd(x: np.ndarray, nper: int) -> tuple:
    """Two-sided Welch PSD sorted by frequency."""
    f, p = sig.welch(x, fs=1.0, nperseg=nper, noverlap=nper // 2,
                     return_onesided=False, detrend=False)
    order = np.argsort(f)
    return f[order], np.asarray(p[order], dtype=np.float64)


def _reference_running_quantile(level_db: np.ndarray, window: int,
                                quantile: int = 25) -> np.ndarray:
    """Order-statistic CFAR reference: a running low quantile.

    Tracks an arbitrarily shaped noise floor, but only while every
    emission is narrower than roughly ``quantile/100`` of the window - a
    wider one lifts its own reference, the middle of the emission drops
    back under threshold, and the detector returns the emission's two
    shaping skirts as separate "signals".
    """
    n = len(level_db)
    window = int(np.clip(int(window) | 1, 9, max(9, (n - 1) | 1)))
    return ndimage.percentile_filter(level_db, quantile, size=window,
                                     mode="nearest")


def _reference_smooth_continuum(level_db: np.ndarray, degree: int = 8,
                                iterations: int = 6,
                                kappa: float = 2.0) -> np.ndarray:
    """Low-order continuum fit with one-sided outlier rejection.

    The complementary failure mode to the running quantile: this model is
    immune to how much of the band an emission covers (the emission is
    rejected as a positive outlier and the floor is fitted underneath
    it), at the price of only representing a smooth receiver passband.
    It is what makes an ordinary RRC burst filling a sixth of a short
    capture come back as ONE box instead of two skirts.
    """
    n = len(level_db)
    if n < 4 * (degree + 1):
        return np.full(n, float(np.percentile(level_db, 25)))
    from numpy.polynomial import chebyshev
    u = np.linspace(-1.0, 1.0, n)
    # Start from the lower 60% of the bins rather than from all of them.
    # A first fit through everything sits inside a wide emission, which
    # makes the residual distribution bimodal, inflates the robust scale,
    # and stops the clipping from rejecting anything at all - the fit
    # then tracks the emission it was supposed to fit underneath.
    keep = level_db <= float(np.percentile(level_db, 60))
    if keep.sum() < 4 * (degree + 1):
        keep = np.ones(n, dtype=bool)
    coef = chebyshev.chebfit(u[keep], level_db[keep], degree)
    for _ in range(max(1, iterations)):
        coef = chebyshev.chebfit(u[keep], level_db[keep], degree)
        resid = level_db - chebyshev.chebval(u, coef)
        base = resid[keep]
        scale = 1.4826 * float(np.median(np.abs(base - np.median(base))))
        scale = max(scale, 1e-6)
        new_keep = resid < kappa * scale
        if new_keep.sum() < max(8, n // 10):
            break
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return chebyshev.chebval(u, coef)


def _reference_models(level_db: np.ndarray) -> list:
    """The candidate noise-floor models, each labelled.

    No single model is right for every capture and the failure mode of
    each is the success of another, so all of them are tried and the
    segmentation objective in :func:`_segmentation_score` chooses.  That
    is a search over one hyperparameter against an explicit objective -
    the same "compact, high-contrast box" criterion the segments
    themselves are ranked by - not a stack of heuristics.
    """
    n = len(level_db)
    out = []
    for k, name in ((32, "quantile, band/32 window"),
                    (8, "quantile, band/8 window"),
                    (2, "quantile, band/2 window")):
        win = max(33, n // k)
        if win >= n:
            continue
        out.append((f"running {name}",
                    _reference_running_quantile(level_db, win)))
    out.append(("smooth continuum fit",
                _reference_smooth_continuum(level_db)))
    out.append(("global 25th percentile",
                np.full(n, float(np.percentile(level_db, 25)))))
    return out


def _bands_from_reference(f: np.ndarray, p: np.ndarray, lvl: np.ndarray,
                          ref: np.ndarray, threshold_db: float,
                          min_bandwidth_bins: int, positive_only: bool,
                          nper: int) -> list:
    """Threshold one PSD against one noise-floor model and label bands."""
    mask = lvl > ref + threshold_db
    if positive_only:
        mask = mask & (f >= 0)
    # close gaps up to ~0.4% of the band (pilot nulls, notches), then
    # discard single-bin spikes
    close = max(3, len(lvl) // 256)
    mask = ndimage.binary_closing(mask, np.ones(close, bool))
    mask = ndimage.binary_opening(mask, np.ones(3, bool))

    nl = 10.0 ** (ref / 10.0)
    excess = np.clip(p - nl, 0.0, None)
    total_excess = float(excess.sum())
    bands = []
    labels, _n = ndimage.label(mask)
    for sl in ndimage.find_objects(labels):
        seg = sl[0]
        if seg.stop - seg.start < max(2, min_bandwidth_bins):
            continue
        lo = float(f[seg.start])
        hi = float(f[min(seg.stop, len(f) - 1)])
        bw = hi - lo
        if bw <= 0:
            continue
        band_excess = float(excess[seg].sum())
        band_noise = float(nl[seg].mean())
        band_power = float(p[seg].mean())
        snr = 10 * np.log10(max(band_power - band_noise, 1e-20) /
                            max(band_noise, 1e-30))
        frac = band_excess / max(total_excess, 1e-30)
        # spectral density gain: power share divided by bandwidth share.
        # The whole band gives 0 dB; a box holding 90% of the power in 4%
        # of the band gives 13.5 dB.
        gain = 10 * np.log10(max(frac, 1e-9) / max(bw, 1e-9))
        # Ranking score: 10*log10(power_fraction^3 / bandwidth).  Density
        # gain alone is maximised by an arbitrarily narrow spike holding
        # a thousandth of the power, so the captured power share has to
        # weigh heavily against the bandwidth.  The cube is not arbitrary:
        # it is the smallest integer power for which the score prefers ONE
        # box around a two-tone FSK emission over either tone alone
        # (tones at +-Rs/2 hold about 45% of the power each in a twentieth
        # of the merged width), while still preferring a tight 2 kHz box
        # to the 16 kHz hull on the AO-73 capture.
        conc = 10 * np.log10(max(frac, 1e-9) ** 3 / max(bw, 1e-9))
        bands.append({"f_low": lo, "f_high": hi, "bandwidth": bw,
                      "snr_db": float(snr), "power_fraction": float(frac),
                      "density_gain_db": float(gain),
                      "concentration_db": float(conc), "nper": int(nper)})
    bands.sort(key=lambda b: -b["concentration_db"])
    return bands


def _segmentations_at_resolution(x: np.ndarray, nper: int,
                                 threshold_db: float,
                                 min_bandwidth_bins: int,
                                 positive_only: bool) -> list:
    """Every (resolution, noise-floor model) segmentation of the band."""
    nper = int(min(nper, max(64, len(x) // 8)))
    if nper < 64 or len(x) < 4 * nper:
        return []
    f, p = _welch_psd(x, nper)
    lvl = 10 * np.log10(p + 1e-30)
    out = []
    for name, ref in _reference_models(lvl):
        bands = _bands_from_reference(f, p, lvl, ref, threshold_db,
                                      min_bandwidth_bins, positive_only,
                                      nper)
        out.append({"bands": bands, "nper": nper, "reference": name,
                    "freq": f, "psd": p, "reference_db": ref,
                    "usable": True})
    return out


def _merge_close_bands(bands: list, merge_gap_ratio: float = 0.25) -> list:
    """Merge boxes separated by less than ``merge_gap_ratio`` of the
    narrower box's width.

    This is what suppresses defect D5 - a clean signal splitting into a
    main box plus a small satellite in its own shaping skirt - without
    fusing genuinely separate emissions.
    """
    if len(bands) < 2:
        return bands
    ordered = sorted(bands, key=lambda b: b["f_low"])
    out = [dict(ordered[0])]
    for b in ordered[1:]:
        prev = out[-1]
        gap = b["f_low"] - prev["f_high"]
        ref_bw = min(prev["bandwidth"], b["bandwidth"])
        if gap <= merge_gap_ratio * ref_bw:
            prev["f_high"] = max(prev["f_high"], b["f_high"])
            prev["bandwidth"] = prev["f_high"] - prev["f_low"]
            prev["power_fraction"] += b["power_fraction"]
            prev["snr_db"] = max(prev["snr_db"], b["snr_db"])
            prev["density_gain_db"] = 10 * np.log10(
                max(prev["power_fraction"], 1e-9) /
                max(prev["bandwidth"], 1e-9))
            prev["concentration_db"] = 10 * np.log10(
                max(prev["power_fraction"], 1e-9) ** 3 /
                max(prev["bandwidth"], 1e-9))
        else:
            out.append(dict(b))
    out.sort(key=lambda b: -b["concentration_db"])
    return out


def _prune_skirts(bands: list, skirt_ratio: float = 0.02,
                  reach: float = 1.0) -> tuple:
    """Drop boxes that are faint satellites inside a much stronger box's
    shaping skirts (defect D5), keeping a record of what was removed."""
    if len(bands) < 2:
        return bands, []
    strongest = max(bands, key=lambda b: b["power_fraction"])
    kept, dropped = [], []
    for b in bands:
        if b is strongest:
            kept.append(b)
            continue
        near = (b["f_low"] > strongest["f_low"] -
                reach * strongest["bandwidth"] and
                b["f_high"] < strongest["f_high"] +
                reach * strongest["bandwidth"])
        if near and b["power_fraction"] < skirt_ratio * \
                strongest["power_fraction"]:
            dropped.append(b)
        else:
            kept.append(b)
    return kept, dropped


def _hull_band(bands: list, min_share: float = 0.05) -> dict:
    """One box spanning every band that carries a real share of the power.

    Some emissions are not contiguous in frequency: an FSK signal is a
    set of tones, an OFDM signal a comb of subcarriers, and a band list
    describes their PARTS.  The parts are legitimate candidates - a tone
    really is a compact, high-contrast box - but so is the emission they
    belong to, and only a downstream demodulation can settle which is
    wanted.  The hull is therefore added to the ranked pool rather than
    replacing anything, so the pipeline can fall back to it.
    """
    strong = [b for b in bands if b["power_fraction"] >= min_share]
    if len(strong) < 2:
        return None
    lo = min(b["f_low"] for b in strong)
    hi = max(b["f_high"] for b in strong)
    bw = hi - lo
    frac = float(min(1.0, sum(b["power_fraction"] for b in strong)))
    if bw <= 0:
        return None
    return {"f_low": lo, "f_high": hi, "bandwidth": bw,
            "snr_db": float(max(b["snr_db"] for b in strong)),
            "power_fraction": frac,
            "density_gain_db": 10 * np.log10(max(frac, 1e-9) / bw),
            "concentration_db": 10 * np.log10(max(frac, 1e-9) ** 3 / bw),
            "nper": strong[0]["nper"], "is_hull": True,
            "n_parts": len(strong)}


def _segmentation_score(bands: list) -> float:
    """Score a whole segmentation: the best box's density gain, penalised
    for shattering the band into many boxes.

    A resolution so fine that noise breaks into hundreds of slivers must
    not win just because one sliver happens to look dense.
    """
    if not bands:
        return -1e9
    best = max(b["concentration_db"] for b in bands)
    fragmentation = 3.0 * np.log10(max(1, len(bands)))
    return float(best - fragmentation)


def _time_extent(wf: dict, f_low: float, f_high: float,
                 margin_db: float = 6.0) -> tuple:
    """Start/end sample of a band's activity, from the waterfall energy
    inside that band."""
    W = wf["waterfall_db"]
    freqs = wf["freq_norm"]
    starts = wf["row_start_sample"]
    if W.size == 0 or len(starts) == 0:
        return 0, 0
    sel = (freqs >= f_low) & (freqs <= f_high)
    if sel.sum() < 1:
        sel = np.zeros(len(freqs), bool)
        sel[int(np.argmin(np.abs(freqs - 0.5 * (f_low + f_high))))] = True
    env = 10 * np.log10((10.0 ** (W[:, sel] / 10.0)).mean(axis=1) + 1e-30)
    if len(env) < 3:
        return int(starts[0]), int(starts[-1] + wf["nfft"])
    floor = float(np.percentile(env, 20))
    active = np.nonzero(env > floor + margin_db)[0]
    if not len(active):
        active = np.arange(len(env))
    a, b = int(active[0]), int(active[-1])
    return int(starts[a]), int(min(starts[b] + wf["nfft"],
                                   starts[-1] + wf["nfft"]))


def _carrier_cross_check(x: np.ndarray, max_samples: int = 1 << 18) -> dict:
    """Locate a suppressed carrier from the x^2 line (report §10.5)."""
    n = min(len(x), max_samples)
    if n < 4096:
        return {"carrier_norm": None, "quality": 0.0}
    z = (np.asarray(x[:n], dtype=np.complex128) ** 2) * np.hanning(n)
    Z = np.abs(np.fft.fft(z))
    Z[0] = 0.0
    med = float(np.median(Z)) + 1e-30
    k = int(np.argmax(Z))
    return {"carrier_norm": float(np.fft.fftfreq(n)[k] / 2.0),
            "line_norm": float(np.fft.fftfreq(n)[k]),
            "quality": float(Z[k] / med)}


def detect_signals(x: np.ndarray, config, sample_rate=None,
                   positive_only: bool = False) -> tuple:
    """Returns (ranked list[SignalSegment], debug dict with waterfall etc.).

    ``positive_only`` is set by the pipeline for a recording that came
    from a single real-valued channel: its spectrum is conjugate
    symmetric, so the negative half carries no independent information
    and would otherwise double every box.
    """
    x = np.asarray(x)
    wf = compute_waterfall(x, nfft=min(config.nfft, 4096), min_rows=16)
    if wf["waterfall_db"].size == 0 or len(x) < 1024:
        return [], {"waterfall": wf, "segmentations": []}

    ladder = [n for n in _RESOLUTION_LADDER if 4 * n <= len(x)]
    if not ladder:
        ladder = [max(64, 1 << int(np.log2(max(64, len(x) // 8))))]
    segmentations = []
    for nper in ladder:
        for seg in _segmentations_at_resolution(
                x, nper, config.threshold_db, config.min_bandwidth_bins,
                positive_only):
            # The grouping scale is searched too.  A multi-tone emission
            # (FSK, an OFDM subcarrier comb) is one signal whose parts are
            # separated by many times their own width, while two genuinely
            # separate emissions look identical at the level of a band
            # list - only the objective can tell them apart, so every
            # grouping is offered to it.
            for ratio in config.merge_gap_ratios:
                variant = dict(seg)
                variant["bands"] = _merge_close_bands(seg["bands"], ratio)
                variant["bands"], variant["skirts_dropped"] = _prune_skirts(
                    variant["bands"], config.skirt_power_ratio)
                variant["merge_gap_ratio"] = ratio
                variant["score"] = _segmentation_score(variant["bands"])
                segmentations.append(variant)
    if not segmentations:
        return [], {"waterfall": wf, "segmentations": []}

    best_seg = max(segmentations, key=lambda s: s["score"])

    # Candidate pool: the winning segmentation's boxes, plus any box from
    # another resolution that is materially denser than everything the
    # winner found.  Resolution disagreement is information, not noise -
    # a burst and a continuous narrowband carrier are best seen at
    # different resolutions and both belong in the ranked list.
    pool = [dict(b, source="best") for b in best_seg["bands"]]
    hull = _hull_band(best_seg["bands"])
    if hull is not None and not any(
            abs(hull["f_low"] - b["f_low"]) < 1e-9 and
            abs(hull["f_high"] - b["f_high"]) < 1e-9 for b in pool):
        pool.append(dict(hull, source="hull"))
    best_gain = max((b["concentration_db"] for b in pool), default=-1e9)
    for seg in segmentations:
        if seg is best_seg or seg["nper"] == best_seg["nper"]:
            continue
        for b in seg["bands"][:3]:
            if b["concentration_db"] < best_gain - 3.0:
                continue
            overlap = any(not (b["f_high"] <= q["f_low"] or
                               b["f_low"] >= q["f_high"]) and
                          abs(b["bandwidth"] - q["bandwidth"]) <
                          0.5 * max(b["bandwidth"], q["bandwidth"])
                          for q in pool)
            if not overlap:
                pool.append(dict(b, source="alt_resolution"))

    # --- independent carrier cross-check -------------------------------
    cross = _carrier_cross_check(x)
    notes_global = []
    fc = cross.get("carrier_norm")
    if fc is not None and cross["quality"] > 20:
        inside = any(b["f_low"] <= fc <= b["f_high"] and
                     b["bandwidth"] < 0.5 for b in pool)
        if not inside:
            # synthesise a box around the x^2 carrier, as wide as the
            # narrowest credible box we already have
            widths = [b["bandwidth"] for b in pool if b["bandwidth"] < 0.5]
            bw = min(widths) if widths else 0.05
            notes_global.append(
                f"x^2 carrier line at {fc:+.5f} cycles/sample "
                f"(quality {cross['quality']:.0f}) lies outside every "
                "detected box; a candidate segment was synthesised around "
                "it")
            pool.append({"f_low": fc - bw / 2, "f_high": fc + bw / 2,
                         "bandwidth": bw, "snr_db": 0.0,
                         "power_fraction": 0.0,
                         "density_gain_db": best_gain - 6.0,
                         "concentration_db": best_gain - 6.0,
                         "nper": 0, "source": "x2_carrier_line"})

    segments = []
    for b in sorted(pool, key=lambda q: -q["concentration_db"]):
        start, end = _time_extent(wf, b["f_low"], b["f_high"])
        if end <= start:
            start, end = 0, len(x)
        rank_score = b["concentration_db"] + 0.25 * b["snr_db"]
        notes = []
        if b.get("source") == "alt_resolution":
            notes.append(f"found only at analysis resolution {b['nper']}")
        elif b.get("source") == "hull":
            notes.append(
                f"hull of {b.get('n_parts', 0)} detected parts: the whole "
                "emission if those parts are one signal (FSK tones, an "
                "OFDM subcarrier comb) rather than separate carriers")
        elif b.get("source") == "x2_carrier_line":
            notes.append("synthesised from the x^2 carrier line, not from "
                         "a threshold crossing")
        segments.append(SignalSegment(
            start_sample=int(start), end_sample=int(end),
            f_low_norm=float(b["f_low"]), f_high_norm=float(b["f_high"]),
            snr_db=float(b["snr_db"]),
            confidence=float(min(1.0, max(0.0, b["snr_db"] / 20.0))),
            sample_rate=sample_rate, id=0,
            method=("order-statistic CFAR on the Welch PSD"
                    if b.get("source") != "x2_carrier_line"
                    else "x^2 carrier line"),
            resolution_bins=int(b["nper"]),
            power_fraction=float(b["power_fraction"]),
            density_gain_db=float(b["density_gain_db"]),
            rank_score=float(rank_score), notes=notes))
        if len(segments) >= config.max_signals:
            break

    segments.sort(key=lambda s: -s.rank_score)
    for k, s in enumerate(segments):
        s.id = k
    if segments and notes_global:
        segments[0].notes.extend(notes_global)

    dbg = {"waterfall": wf,
           "chosen_resolution": int(best_seg["nper"]),
           "chosen_noise_model": best_seg["reference"],
           "chosen_merge_gap_ratio": best_seg.get("merge_gap_ratio"),
           "resolutions_tried": sorted({int(s["nper"])
                                        for s in segmentations}),
           "segmentation_scores": [
               {"resolution_bins": int(s["nper"]),
                "noise_model": s["reference"],
                "merge_gap_ratio": s.get("merge_gap_ratio"),
                "n_bands": len(s["bands"]),
                "score": round(s["score"], 2)}
               for s in sorted(segmentations, key=lambda q: -q["score"])[:12]],
           "skirts_dropped": len(best_seg.get("skirts_dropped", [])),
           "carrier_cross_check": cross,
           "noise_floor_db": best_seg.get("reference_db"),
           "notes": notes_global}
    return segments, dbg
