import os

import numpy as np

from rfanalyzer.ingestion import load_recording, sniff_raw_iq
from rfanalyzer.ingestion.wav import read_wav


def _tone(n=20000, f=0.1):
    t = np.arange(n)
    return (np.exp(2j * np.pi * f * t) +
            0.05 * (np.random.default_rng(0).normal(size=n) +
                    1j * np.random.default_rng(1).normal(size=n))).astype(np.complex64)


def test_raw_iq_complex64(tmp_path):
    p = str(tmp_path / "a.iq")
    _tone().tofile(p)
    rec = load_recording(p)
    assert rec.datatype == "complex64"
    assert rec.sample_rate is None
    assert any("sample rate unknown" in w for w in rec.warnings)
    assert rec.samples.dtype == np.complex64


def test_raw_iq_int16_sniff(tmp_path):
    p = str(tmp_path / "b.iq")
    x = _tone()
    interleaved = np.empty(2 * len(x), dtype=np.int16)
    interleaved[0::2] = (x.real * 8000).astype(np.int16)
    interleaved[1::2] = (x.imag * 8000).astype(np.int16)
    interleaved.tofile(p)
    cands = sniff_raw_iq(p)
    assert cands[0]["dtype"] == "int16"
    rec = load_recording(p, sample_rate=48000)
    assert rec.sample_rate == 48000
    assert rec.datatype == "int16"


def test_wav_stereo_iq(tmp_path):
    import scipy.io.wavfile as wavfile
    p = str(tmp_path / "c.wav")
    x = _tone()
    stereo = np.stack([(x.real * 20000).astype(np.int16),
                       (x.imag * 20000).astype(np.int16)], axis=1)
    wavfile.write(p, 96000, stereo)
    rec = load_recording(p)
    assert rec.format == "wav"
    assert rec.sample_rate == 96000
    assert rec.channels == 2
    # spectrum should peak near +0.1 normalised
    X = np.abs(np.fft.fft(rec.samples[:8192]))
    peak = np.fft.fftfreq(8192)[np.argmax(X)]
    assert abs(peak - 0.1) < 0.01


def test_sigmf_sidecar(tmp_path):
    import json
    p = str(tmp_path / "d.sigmf-data")
    _tone().tofile(p)
    with open(str(tmp_path / "d.sigmf-meta"), "w") as f:
        json.dump({"global": {"core:datatype": "cf32_le",
                              "core:sample_rate": 250000.0},
                   "captures": [{"core:sample_start": 0,
                                 "core:frequency": 14.1e6}]}, f)
    rec = load_recording(p)
    assert rec.sample_rate == 250000.0
    assert rec.center_frequency == 14.1e6
