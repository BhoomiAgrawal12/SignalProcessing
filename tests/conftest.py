import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dhwani.common.config import Config


@pytest.fixture(scope="session")
def config():
    cfg = Config()
    ckpt = os.path.join(os.path.dirname(__file__), "..", "ml", "cvnet_rf",
                        "checkpoints", "real_best.pt")
    if os.path.exists(ckpt):
        cfg.modulation.cvnet_checkpoint = os.path.abspath(ckpt)
    else:
        cfg.modulation.cvnet_enabled = False
    return cfg


@pytest.fixture()
def rng():
    return np.random.default_rng(1234)
