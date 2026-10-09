"""1.H: CLI detect/classify run the same Analyzer engine as analyze."""
import json

from dhwani import cli
from dhwani.pipeline import Analyzer


def test_classify_matches_analyze(tmp_path, config, capsys):
    f = str(tmp_path / "q.iq")
    assert cli.main(["synth", f, "--modulation", "QPSK", "--frames", "20",
                     "--seed", "3"]) == 0
    capsys.readouterr()
    assert cli.main(["classify", f, "--json", "--no-ml", "--no-cache"]) == 0
    got = json.loads(capsys.readouterr().out)
    res = Analyzer(config, use_cache=False).analyze(f, no_ml=True,
                                                      stop_after="S5")
    assert got["modulation"]["prediction"] == res.modulation.prediction
    assert got["parameters"]["symbol_rate_norm"] == \
        res.parameters.symbol_rate_norm
    s6 = next(t for t in res.pipeline_trace if t["stage"] == "S6")
    assert s6["status"] == "skipped"

    assert cli.main(["detect", f, "--json", "--no-cache"]) == 0
    det = json.loads(capsys.readouterr().out)
    assert len(det["signals"]) >= 1
