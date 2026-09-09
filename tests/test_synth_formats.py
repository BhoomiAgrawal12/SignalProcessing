"""Synthetic output formats: raw IQ, WAV, SigMF - generation + ingestion
round-trips, plus one full end-to-end recovery through a WAV file."""
import json
import logging

import numpy as np
import pytest

from rfanalyzer.ingestion import load_recording
from rfanalyzer.pipeline import RFAnalyzer
from rfanalyzer.synth.factory import WaveformFactory
from rfanalyzer.synth.writers import write_iq, write_wav, write_sigmf

logging.disable(logging.INFO)


@pytest.fixture(scope="module")
def signal():
    fac = WaveformFactory(seed=5)
    return fac.generate(modulation="QPSK", sps=8.0, snr_db=25.0,
                        cfo_norm=0.004, n_frames=80)


def test_iq_roundtrip(tmp_path, signal, config):
    iq, gt = signal
    p = str(tmp_path / "a.iq")
    out = write_iq(iq, gt, p, sample_rate=1e6)
    truth = json.load(open(out["truth"]))
    assert truth["modulation"] == "QPSK"
    rec = load_recording(p, sample_rate=1e6)
    assert np.allclose(rec.samples[:1000], iq[:1000])


def test_wav_roundtrip(tmp_path, signal):
    iq, gt = signal
    p = str(tmp_path / "a.wav")
    write_wav(iq, gt, p, sample_rate=1e6)
    rec = load_recording(p)
    assert rec.format == "wav"
    assert rec.sample_rate == 1e6
    assert rec.channels == 2
    a, b = rec.samples[:4000], iq[:4000]
    corr = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert corr > 0.9999          # float32 stereo is exact up to scaling


def test_wav_requires_sample_rate(tmp_path, signal):
    iq, gt = signal
    with pytest.raises(ValueError):
        write_wav(iq, gt, str(tmp_path / "b.wav"), sample_rate=None)


def test_sigmf_roundtrip(tmp_path, signal):
    iq, gt = signal
    out = write_sigmf(iq, gt, str(tmp_path / "a"), sample_rate=1e6,
                      center_frequency=433.92e6)
    meta = json.load(open(out["meta"]))
    assert meta["global"]["core:datatype"] == "cf32_le"
    assert meta["global"]["core:sample_rate"] == 1e6
    assert meta["captures"][0]["core:frequency"] == 433.92e6
    assert meta["global"]["rfanalyzer:ground_truth"]["modulation"] == "QPSK"
    rec = load_recording(out["data"])
    assert rec.sample_rate == 1e6
    assert rec.center_frequency == 433.92e6
    assert np.allclose(rec.samples[:1000], iq[:1000])


@pytest.mark.slow
def test_full_e2e_through_wav(tmp_path, config):
    """generate -> write WAV -> ingest -> S0-S11 -> payload recovered."""
    fac = WaveformFactory(seed=99)
    iq, gt = fac.generate(
        modulation="QPSK", sps=8.0, snr_db=22.0, cfo_norm=0.006,
        phase_offset=0.5,
        fec={"family": "convolutional", "K": 7, "generators": (0o171, 0o133)},
        interleaver={"kind": "block", "rows": 8, "cols": 16},
        scrambler={"kind": "known_whitening", "name": "PN9-CC1101"},
        n_frames=80)
    p = str(tmp_path / "full.wav")
    write_wav(iq, gt, p, sample_rate=1e6)
    res = RFAnalyzer(config, use_cache=False).analyze(p)
    assert res.recording_meta["sample_rate"] == 1e6
    assert res.modulation.prediction == "QPSK"
    assert res.fec.family == "convolutional"
    assert res.frames.crc["name"] == "CRC-16-CCITT-FALSE"
    assert res.payload_intelligence["available"]
    assert res.payload_intelligence["provenance"]["crc_validated"]


@pytest.mark.slow
def test_e2e_ldpc(tmp_path, config):
    """LDPC-coded signal identified and decoded through the pipeline."""
    fac = WaveformFactory(seed=21)
    iq, gt = fac.generate(modulation="QPSK", sps=8.0, snr_db=25.0,
                          cfo_norm=0.003,
                          fec={"family": "ldpc", "n": 256, "k": 128, "seed": 1},
                          n_frames=80)
    p = str(tmp_path / "ldpc.iq")
    iq.astype(np.complex64).tofile(p)
    res = RFAnalyzer(config, use_cache=False).analyze(p, sample_rate=1e6)
    assert res.fec is not None and res.fec.family == "ldpc"
    assert res.fec.parameters["n"] == 256
    assert res.fec.syndrome_zero_rate > 0.9
