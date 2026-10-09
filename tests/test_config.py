import json
import os

from dhwani.common.config import Config

DEFAULT = os.path.join(os.path.dirname(__file__), "..", "configs", "default.json")


def test_default_json_sets_only_known_keys():
    """2.A: every key in the shipped config maps to a field something reads."""
    cfg = Config()
    with open(DEFAULT) as f:
        data = json.load(f)
    for sec, val in data.items():
        assert hasattr(cfg, sec), sec
        if isinstance(val, dict) and sec != "fusion_weights":
            for k in val:
                assert hasattr(getattr(cfg, sec), k), f"{sec}.{k}"


def test_unknown_key_warns(tmp_path, caplog):
    """2.E: unknown keys are ignored *with a warning*, as documented."""
    from dhwani.common.config import load_config
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"cfar": {"guard_bins": 2, "nfft": 2048},
                             "bogus": 1}))
    cfg = load_config(str(p))
    assert cfg.cfar.nfft == 2048 and not hasattr(cfg.cfar, "guard_bins")
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "cfar.guard_bins" in msgs and "bogus" in msgs
