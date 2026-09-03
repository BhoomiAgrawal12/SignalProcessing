"""End-to-end regression tests over the synthetic waveform factory.

Each test generates a signal with full known ground truth, writes it as a
raw .iq file, and checks the blind pipeline recovers the structure."""
import logging

import numpy as np
import pytest

from rfanalyzer.pipeline import RFAnalyzer
from rfanalyzer.synth.factory import WaveformFactory

logging.disable(logging.INFO)


def _run(tmp_path, config, gen_kwargs, sample_rate=1e6):
    fac = WaveformFactory(seed=99)
    iq, gt = fac.generate(**gen_kwargs)
    p = str(tmp_path / "sig.iq")
    iq.astype(np.complex64).tofile(p)
    an = RFAnalyzer(config, use_cache=False)
    return an.analyze(p, sample_rate=sample_rate), gt


def test_e2e_full_stack_qpsk(tmp_path, config):
    """QPSK + conv FEC + block interleaver + PN9 + framed CRC payload."""
    res, gt = _run(tmp_path, config, dict(
        modulation="QPSK", sps=8.0, snr_db=22.0, cfo_norm=0.006,
        phase_offset=0.5,
        fec={"family": "convolutional", "K": 7, "generators": (0o171, 0o133)},
        interleaver={"kind": "block", "rows": 8, "cols": 16},
        scrambler={"kind": "known_whitening", "name": "PN9-CC1101"},
        n_frames=80))
    assert res.modulation.prediction == "QPSK"
    assert abs(res.parameters.symbol_rate_norm - 0.125) < 0.002
    assert res.scrambler.name == "PN9-CC1101"
    assert res.interleaver.kind == "block"
    assert (res.interleaver.parameters["rows"],
            res.interleaver.parameters["cols"]) == (8, 16)
    assert res.fec.family == "convolutional"
    assert res.fec.parameters["K"] == 7
    assert res.fec.syndrome_zero_rate > 0.98
    assert res.frames.frame_length_bits == 96
    assert res.frames.sync_word_hex.startswith("eb90")
    assert res.frames.crc["name"] == "CRC-16-CCITT-FALSE"
    assert res.frames.crc["pass_fraction"] > 0.9


def test_e2e_plain_bpsk(tmp_path, config):
    """Uncoded, unscrambled BPSK: verdicts must honestly be 'none'."""
    res, gt = _run(tmp_path, config, dict(
        modulation="BPSK", sps=8.0, snr_db=18.0, cfo_norm=0.004,
        n_frames=80))
    assert res.modulation.prediction == "BPSK"
    assert res.interleaver is None or res.interleaver.kind in ("none",)
    assert res.fec is None or res.fec.family == "none"
    assert res.frames is not None and res.frames.frame_length_bits == 96
    assert res.frames.crc is not None
    assert res.frames.crc["pass_fraction"] > 0.9


def test_e2e_2fsk(tmp_path, config):
    res, gt = _run(tmp_path, config, dict(
        modulation="2FSK", sps=8.0, snr_db=17.0, n_frames=80))
    assert res.modulation.prediction == "2FSK"
    assert res.frames is not None
    assert res.frames.frame_length_bits == 96


def test_e2e_rs_coded(tmp_path, config):
    res, gt = _run(tmp_path, config, dict(
        modulation="QPSK", sps=8.0, snr_db=25.0, cfo_norm=0.003,
        fec={"family": "reed_solomon", "n": 255, "k": 223},
        n_frames=60))
    assert res.modulation.prediction == "QPSK"
    assert res.fec is not None and res.fec.family == "reed_solomon"
    assert res.fec.parameters["n"] == 255


def test_e2e_noise_only(tmp_path, config):
    rng = np.random.default_rng(5)
    noise = (rng.normal(size=40000) + 1j * rng.normal(size=40000)).astype(np.complex64)
    p = str(tmp_path / "noise.iq")
    noise.tofile(p)
    an = RFAnalyzer(config, use_cache=False)
    res = an.analyze(p, sample_rate=1e6)
    # noise may produce marginal detections, but never confident ones
    assert all(s.confidence < 0.5 for s in res.segments)
    if res.modulation is not None:
        assert res.modulation.confidence < 0.9 or \
            res.modulation.prediction == "UNKNOWN"
