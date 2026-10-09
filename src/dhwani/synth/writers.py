"""Format writers for the synthetic factory.

One canonical complex64 IQ array feeds every writer, so all formats
represent exactly the same signal:

    WaveformFactory.generate() -> complex64 IQ + GroundTruth
        -> write_iq     x.iq          (raw interleaved complex64)
        -> write_wav    x.wav         (stereo: left = I, right = Q)
        -> write_sigmf  x.sigmf-data + x.sigmf-meta (cf32_le)

Every writer also emits <base>_truth.json with the full ground truth,
matching the convention the regression tests already rely on.
"""
from __future__ import annotations

import json
import os
import struct
from typing import Optional

import numpy as np

from .factory import GroundTruth

SIGMF_VERSION = "1.0.0"


def _write_truth(base: str, gt: GroundTruth, sample_rate: Optional[float]):
    truth = json.loads(gt.to_json())
    if sample_rate:
        truth["sample_rate"] = sample_rate
        truth["symbol_rate_hz"] = gt.symbol_rate_norm * sample_rate
    path = base + "_truth.json"
    with open(path, "w") as f:
        json.dump(truth, f, indent=2)
    return path


def write_iq(iq: np.ndarray, gt: GroundTruth, path: str,
             sample_rate: Optional[float] = None) -> dict:
    """Raw interleaved complex64, the existing default format."""
    iq.astype(np.complex64).tofile(path)
    base = os.path.splitext(path)[0]
    return {"data": path, "truth": _write_truth(base, gt, sample_rate)}


def write_wav(iq: np.ndarray, gt: GroundTruth, path: str,
              sample_rate: float, bits: int = 32) -> dict:
    """Stereo WAV: left channel = I, right channel = Q.

    Default is IEEE float32 (audio format 3), which round-trips the
    samples exactly and is read natively by the S0 ingest stage; bits=16
    writes PCM16 for tools that cannot handle float WAVs.
    """
    if not sample_rate:
        raise ValueError("WAV output requires --sample-rate (the value is "
                         "written into the RIFF fmt header)")
    x = np.asarray(iq, dtype=np.complex64)
    peak = float(max(np.abs(x.real).max(), np.abs(x.imag).max()) or 1.0)
    scale = 0.9 / peak            # headroom so nothing clips in PCM
    if bits == 32:
        frames = np.empty((len(x), 2), dtype=np.float32)
        frames[:, 0] = (x.real * scale).astype(np.float32)
        frames[:, 1] = (x.imag * scale).astype(np.float32)
        audio_format, sample_bytes = 3, 4
        raw = frames.tobytes()
    elif bits == 16:
        frames = np.empty((len(x), 2), dtype=np.int16)
        frames[:, 0] = np.round(x.real * scale * 32767).astype(np.int16)
        frames[:, 1] = np.round(x.imag * scale * 32767).astype(np.int16)
        audio_format, sample_bytes = 1, 2
        raw = frames.tobytes()
    else:
        raise ValueError("bits must be 16 or 32")
    n_channels = 2
    byte_rate = int(sample_rate) * n_channels * sample_bytes
    block_align = n_channels * sample_bytes
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(raw)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, audio_format, n_channels,
                            int(sample_rate), byte_rate, block_align,
                            sample_bytes * 8))
        f.write(b"data")
        f.write(struct.pack("<I", len(raw)))
        f.write(raw)
    base = os.path.splitext(path)[0]
    return {"data": path, "truth": _write_truth(base, gt, sample_rate)}


def write_sigmf(iq: np.ndarray, gt: GroundTruth, base_path: str,
                sample_rate: Optional[float] = None,
                center_frequency: Optional[float] = None) -> dict:
    """SigMF v1 pair: <base>.sigmf-data (cf32_le) + <base>.sigmf-meta.

    Standard information goes in SigMF core fields; the synthetic ground
    truth lives in the project extension namespace `dhwani:` so the
    files stay schema-valid for other SigMF tools.
    """
    if base_path.endswith((".sigmf", ".sigmf-data", ".sigmf-meta")):
        base_path = base_path.rsplit(".sigmf", 1)[0]
    data_path = base_path + ".sigmf-data"
    meta_path = base_path + ".sigmf-meta"
    iq.astype(np.complex64).tofile(data_path)
    capture = {"core:sample_start": 0}
    if center_frequency:
        capture["core:frequency"] = float(center_frequency)
    meta = {
        "global": {
            "core:version": SIGMF_VERSION,
            "core:datatype": "cf32_le",
            "core:description": "Dhwani synthetic waveform "
                                "(ground truth in dhwani namespace)",
            **({"core:sample_rate": float(sample_rate)} if sample_rate else {}),
            "dhwani:ground_truth": json.loads(gt.to_json()),
        },
        "captures": [capture],
        "annotations": [{
            "core:sample_start": 0,
            "core:sample_count": int(len(iq)),
            "core:label": f"synthetic {gt.modulation}",
        }],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"data": data_path, "meta": meta_path,
            "truth": _write_truth(base_path, gt, sample_rate)}


def write_recording(iq: np.ndarray, gt: GroundTruth, path: str,
                    sample_rate: Optional[float] = None,
                    center_frequency: Optional[float] = None,
                    wav_bits: int = 32) -> dict:
    """Dispatch on the output extension: .iq/.bin/.raw, .wav, .sigmf*."""
    low = path.lower()
    if low.endswith(".wav"):
        return write_wav(iq, gt, path, sample_rate, bits=wav_bits)
    if low.endswith((".sigmf", ".sigmf-data", ".sigmf-meta")):
        return write_sigmf(iq, gt, path, sample_rate, center_frequency)
    return write_iq(iq, gt, path, sample_rate)
