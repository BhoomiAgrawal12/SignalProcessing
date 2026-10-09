"""1.B: the CVNet checkpoint is untrusted input; it must never run code."""
import pickle

import pytest

from dhwani.modulation import cvnet


class _Boom:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_sha256_mismatch_raises(tmp_path):
    p = tmp_path / "real_best.pt"
    p.write_bytes(b"not the pinned checkpoint")
    with pytest.raises(ValueError, match="sha256"):
        cvnet.check_sha256(str(p), cvnet.SHA256["real"])


def test_pickle_checkpoint_raises_without_executing(tmp_path):
    pytest.importorskip("torch")
    marker = tmp_path / "pwned"
    p = tmp_path / "evil.pt"
    p.write_bytes(pickle.dumps({"model_state": _Boom(str(marker))}))
    with pytest.raises(Exception):
        cvnet.load_cvnet(str(p), "real", device="cpu")
    assert not marker.exists()
