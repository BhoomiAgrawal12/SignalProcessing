"""A. Byte and bit forensics: statistics only, no interpretation."""
from __future__ import annotations


import numpy as np


def shannon_entropy_bytes(data: bytes) -> float:
    """Entropy in bits per byte (0..8)."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(data)
    return float(-(p * np.log2(p)).sum())


def entropy_windows(data: bytes, window: int = 64) -> list:
    out = []
    for off in range(0, len(data), window):
        chunk = data[off:off + window]
        if len(chunk) >= 16:
            out.append({"offset": off,
                        "entropy": round(shannon_entropy_bytes(chunk) / 8, 3)})
    return out[:512]


def bit_analysis(bits: np.ndarray) -> dict:
    if bits is None or len(bits) == 0:
        return {}
    b = np.asarray(bits, dtype=np.uint8)
    ones = float(b.mean())
    # run lengths
    changes = np.flatnonzero(np.diff(b)) + 1
    edges = np.concatenate([[0], changes, [len(b)]])
    runs = np.diff(edges)
    # periodicity via bit autocorrelation at small lags
    x = b.astype(np.float64) * 2 - 1
    period, best = None, 0.0
    for lag in range(2, min(256, len(b) // 4)):
        r = float(np.mean(x[:-lag] * x[lag:]))
        if r > 0.5 and r > best:
            best, period = r, lag
    return {"ones_ratio": round(ones, 4),
            "longest_zero_run": int(runs[::2].max() if b[0] == 0 else
                                    runs[1::2].max() if len(runs) > 1 else 0),
            "longest_one_run": int(runs[::2].max() if b[0] == 1 else
                                   runs[1::2].max() if len(runs) > 1 else 0),
            "bit_periodicity": ({"lag": period, "correlation": round(best, 3)}
                                if period else None)}


def analyze_bytes(data: bytes, bits: np.ndarray = None,
                  max_bytes: int = 1 << 20) -> dict:
    data = data[:max_bytes]
    if not data:
        return {"n_bytes": 0}
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    top = np.argsort(counts)[::-1][:8]
    printable = float(((arr >= 0x20) & (arr < 0x7F)).mean())
    H = shannon_entropy_bytes(data)
    return {
        "n_bytes": len(data),
        "n_bits": len(bits) if bits is not None else len(data) * 8,
        "entropy_bits_per_byte": round(H, 4),
        "entropy_normalised": round(H / 8, 4),
        "printable_ratio": round(printable, 4),
        "null_ratio": round(float((arr == 0).mean()), 4),
        "control_ratio": round(float(((arr < 0x20) & (arr != 9) &
                                      (arr != 10) & (arr != 13)).mean()), 4),
        "unique_bytes": int((counts > 0).sum()),
        "most_common_bytes": [{"byte": f"0x{b:02x}",
                               "fraction": round(float(counts[b] / len(data)), 4)}
                              for b in top if counts[b] > 0],
        "entropy_windows": entropy_windows(data),
        "bit_analysis": bit_analysis(bits),
    }
