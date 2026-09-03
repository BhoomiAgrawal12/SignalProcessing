"""Stage S2 displays: Welch PSD and STFT waterfall (decimated for display)."""
from __future__ import annotations

import numpy as np
from scipy import signal as sig


def compute_psd(x: np.ndarray, nfft: int = 4096, max_samples: int = 1 << 22) -> dict:
    x = np.asarray(x[:max_samples])
    nper = min(nfft, len(x))
    f, p = sig.welch(x, fs=1.0, nperseg=nper, noverlap=nper // 2,
                     return_onesided=False, detrend=False)
    idx = np.argsort(f)
    return {"freq_norm": f[idx], "psd_db": 10 * np.log10(p[idx] + 1e-20)}


def compute_waterfall(x: np.ndarray, nfft: int = 1024,
                      max_rows: int = 512, max_samples: int = 1 << 24,
                      min_rows: int = 16) -> dict:
    """Time-frequency matrix for display + detection. Rows are averaged
    (decimated) so even huge files produce <= max_rows rows; for short
    recordings the FFT size shrinks so there are at least `min_rows` rows
    of time resolution for the detector to work with."""
    x = np.asarray(x[:max_samples])
    while nfft > 128 and len(x) // nfft < min_rows:
        nfft //= 2
    hop = nfft
    n_rows_full = max(1, (len(x) - nfft) // hop + 1)
    stride = max(1, n_rows_full // max_rows)
    rows = []
    win = np.hanning(nfft).astype(np.float32)
    row_starts = []
    for r in range(0, n_rows_full, stride):
        s = r * hop
        seg = x[s:s + nfft]
        if len(seg) < nfft:
            break
        X = np.fft.fftshift(np.abs(np.fft.fft(seg * win)) ** 2)
        rows.append(10 * np.log10(X + 1e-20))
        row_starts.append(s)
    wf = np.array(rows, dtype=np.float32)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft))
    return {"waterfall_db": wf, "freq_norm": freqs,
            "row_start_sample": np.array(row_starts),
            "nfft": nfft, "hop": hop * stride}
