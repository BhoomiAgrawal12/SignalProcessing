"""Disk cache for expensive stage results.

Key = (recording content hash, stage name, stage version, parameter hash).
Bump a stage's VERSION constant whenever its algorithm changes to invalidate
old entries.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Callable, Optional

import numpy as np


def content_hash(samples: np.ndarray, max_bytes: int = 1 << 20) -> str:
    """Hash of dtype + shape + head/tail slices (full file hash would defeat
    the point of memory mapping for multi-GB recordings)."""
    h = hashlib.sha256()
    h.update(str(samples.dtype).encode())
    h.update(str(samples.shape).encode())
    step = max(1, samples.nbytes // max_bytes)
    view = samples[:: step] if step > 1 else samples
    h.update(np.ascontiguousarray(view[:200000]).tobytes())
    return h.hexdigest()[:24]


def params_hash(params: dict) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]


class StageCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.dir = cache_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, key + ".pkl")

    def key(self, data_hash: str, stage: str, version: int, params: dict) -> str:
        return f"{data_hash}_{stage}_v{version}_{params_hash(params)}"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        p = self._path(key)
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                os.remove(p)
        return None

    def put(self, key: str, value: Any):
        if not self.enabled:
            return
        try:
            with open(self._path(key), "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def get_or_compute(self, key: str, fn: Callable[[], Any]) -> Any:
        v = self.get(key)
        if v is None:
            v = fn()
            self.put(key, v)
        return v
