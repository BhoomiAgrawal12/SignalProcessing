"""Structured logging with per-analysis run IDs and stage timing."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_configured = False


def _configure(level: str = "INFO"):
    global _configured
    if not _configured:
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                            format=_FORMAT)
        _configured = True


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    _configure(level)
    return logging.getLogger(name)


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


class StageTimer:
    """Collects wall-clock time per pipeline stage."""

    def __init__(self, logger: logging.Logger, timings: dict):
        self.logger = logger
        self.timings = timings

    @contextmanager
    def stage(self, name: str, **params):
        t0 = time.perf_counter()
        self.logger.info("stage %s started %s", name,
                         f"params={params}" if params else "")
        try:
            yield
        except Exception:
            self.timings[name] = time.perf_counter() - t0
            self.logger.exception("stage %s FAILED after %.2fs", name,
                                  self.timings[name])
            raise
        else:
            self.timings[name] = time.perf_counter() - t0
            self.logger.info("stage %s finished in %.2fs", name, self.timings[name])
