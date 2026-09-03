"""Unified recording loader (stage S0)."""
from __future__ import annotations

import json
import os

import numpy as np

from ..common.models import Recording
from .sniffer import sniff_raw_iq
from .wav import read_wav, wav_to_complex


def _raw_iq_to_complex(path: str, dtype: str, endian: str,
                       mmap_threshold: int = 64 << 20) -> np.ndarray:
    dt = np.dtype({"complex64": np.complex64, "float32": np.float32,
                   "int16": np.int16, "int8": np.int8,
                   "uint8": np.uint8}[dtype])
    dt = dt.newbyteorder("<" if endian == "little" else ">")
    size = os.path.getsize(path)
    count = size // dt.itemsize
    big = size > mmap_threshold
    raw = np.memmap(path, dtype=dt, mode="r", shape=(count,)) if big \
        else np.fromfile(path, dtype=dt, count=count)
    if dtype == "complex64":
        return raw if big else raw.astype(np.complex64)
    # interleaved I/Q -> complex64 (conversion is lazy for memmaps: we
    # convert in chunks only when slices are requested by later stages;
    # for simplicity here we convert eagerly but in blocks to bound RAM)
    n2 = (count // 2) * 2
    out = np.empty(n2 // 2, dtype=np.complex64)
    block = 1 << 22
    for s in range(0, n2, block):
        e = min(s + block, n2)
        seg = np.asarray(raw[s:e], dtype=np.float32)
        if dtype == "uint8":
            seg = (seg - 127.5) / 127.5
        elif dtype == "int8":
            seg = seg / 128.0
        elif dtype == "int16":
            seg = seg / 32768.0
        out[s // 2:e // 2] = seg[0::2] + 1j * seg[1::2]
    return out


def _find_sigmf_meta(path: str):
    for cand in (path + ".sigmf-meta",
                 os.path.splitext(path)[0] + ".sigmf-meta"):
        if os.path.exists(cand):
            return cand
    return None


def load_recording(path: str, sample_rate: float = None,
                   center_frequency: float = None, datatype: str = None,
                   endian: str = "little") -> Recording:
    """Load .wav / .iq / .sigmf-data into a normalised Recording.

    User-supplied sample_rate/center_frequency/datatype always win over
    sniffing. A missing sample rate is reported as unknown, never guessed.
    """
    ext = os.path.splitext(path)[1].lower()
    warnings = []

    # SigMF sidecar?
    meta_path = _find_sigmf_meta(path)
    sig_meta = {}
    if meta_path:
        try:
            with open(meta_path) as f:
                sm = json.load(f)
            g = sm.get("global", {})
            sig_meta = {"sample_rate": g.get("core:sample_rate"),
                        "datatype": g.get("core:datatype"),
                        "captures": sm.get("captures", [])}
            caps = sig_meta["captures"]
            if caps and "core:frequency" in caps[0]:
                sig_meta["center_frequency"] = caps[0]["core:frequency"]
        except Exception as e:
            warnings.append(f"failed to parse SigMF sidecar: {e}")

    if ext == ".wav":
        meta = read_wav(path)
        samples, is_real = wav_to_complex(meta)
        if is_real:
            warnings.append("single-channel WAV: treated as real signal "
                            "(analytic conversion applied downstream)")
        return Recording(
            file_path=path, format="wav", samples=samples,
            sample_rate=sample_rate or meta["sample_rate"],
            sample_rate_source="user" if sample_rate else "wav_header",
            center_frequency=center_frequency or meta.get("center_frequency"),
            center_frequency_source=("user" if center_frequency else
                                     "auxi_chunk" if meta.get("center_frequency")
                                     else "unknown"),
            datatype=meta.get("datatype", ""), endianness="little",
            channels=meta.get("channels", 1),
            timestamp=meta.get("timestamp"), warnings=warnings)

    # raw IQ (or sigmf-data)
    sniff = [] if datatype else sniff_raw_iq(path)
    if datatype:
        chosen = {"dtype": datatype, "endian": endian, "confidence": 1.0,
                  "explanation": "user-specified"}
    elif sig_meta.get("datatype"):
        dt_map = {"cf32_le": ("complex64", "little"), "ci16_le": ("int16", "little"),
                  "ci8": ("int8", "little"), "cu8": ("uint8", "little"),
                  "cf32_be": ("complex64", "big"), "ci16_be": ("int16", "big")}
        d, e = dt_map.get(sig_meta["datatype"], ("complex64", "little"))
        chosen = {"dtype": d, "endian": e, "confidence": 1.0,
                  "explanation": f"SigMF datatype {sig_meta['datatype']}"}
    elif sniff:
        chosen = sniff[0]
    else:
        raise ValueError(f"cannot determine format of {path}; specify --datatype")

    samples = _raw_iq_to_complex(path, chosen["dtype"], chosen.get("endian", "little"))
    sr = sample_rate or sig_meta.get("sample_rate")
    cf = center_frequency or sig_meta.get("center_frequency")
    if sr is None:
        warnings.append("sample rate unknown (headerless raw IQ): all "
                        "frequencies reported in normalised units; set "
                        "--sample-rate to obtain absolute values")
    return Recording(
        file_path=path, format="sigmf" if sig_meta else "raw_iq",
        samples=samples, sample_rate=sr,
        sample_rate_source=("user" if sample_rate else
                            "sigmf" if sig_meta.get("sample_rate") else "unknown"),
        center_frequency=cf,
        center_frequency_source=("user" if center_frequency else
                                 "sigmf" if sig_meta.get("center_frequency") else "unknown"),
        datatype=chosen["dtype"], endianness=chosen.get("endian", "little"),
        format_confidence=chosen.get("confidence", 1.0),
        sniff_report={"candidates": sniff[:5]} if sniff else {},
        warnings=warnings)
