"""Format-independence and payload ground-truth regression tests.

The same QPSK text-payload recording is written as raw .iq, stereo WAV
and SigMF; the blind pipeline must classify it, validate every frame
CRC, report the true sync word at the frame start and extract a payload
that is byte-for-byte a contiguous substring of the transmitted payload
(blind acquisition legitimately loses frames at the burst edges)."""
import logging

import numpy as np
import pytest

from dhwani.pipeline import Analyzer
from dhwani.synth.factory import WaveformFactory
from dhwani.synth.writers import write_recording

logging.disable(logging.INFO)


@pytest.fixture(scope="module")
def qpsk_text():
    fac = WaveformFactory(seed=7)
    iq, gt = fac.generate(modulation="QPSK", sps=8.0, snr_db=20.0,
                          cfo_norm=0.005, phase_offset=0.4,
                          payload_mode="text", n_frames=80)
    return iq, gt


@pytest.mark.parametrize("ext", ["iq", "wav", "sigmf"])
def test_format_roundtrip_payload(tmp_path, config, qpsk_text, ext):
    iq, gt = qpsk_text
    path = str(tmp_path / f"sig.{ext}")
    write_recording(iq, gt, path, sample_rate=1e6)
    data_path = path + "-data" if ext == "sigmf" else path
    res = Analyzer(config, use_cache=False).analyze(
        data_path, sample_rate=None if ext != "iq" else 1e6)

    assert res.modulation.prediction in ("QPSK", "OQPSK")
    assert res.frames is not None
    assert res.frames.sync_word_hex == gt.frame["sync_hex"]
    assert res.frames.sync_offset == 0
    assert res.frames.crc["pass_fraction"] == 1.0

    truth_payload = bytes.fromhex("".join(gt.payloads))
    extracted = res.payload.data
    assert extracted, "no payload extracted"
    assert extracted in truth_payload, \
        "extracted payload is not a contiguous substring of the truth"
    assert len(extracted) >= len(truth_payload) // 2

    pi = res.payload_intelligence or {}
    assert (pi.get("summary") or {}).get("trust") == "VALIDATED PAYLOAD"


def test_random_payload_match(tmp_path, config):
    """Random (binary-class) payloads must also survive field stripping."""
    fac = WaveformFactory(seed=5)
    iq, gt = fac.generate(modulation="QPSK", sps=8.0, snr_db=20.0,
                          cfo_norm=0.005, phase_offset=0.4, n_frames=80)
    p = str(tmp_path / "sig.iq")
    iq.astype(np.complex64).tofile(p)
    res = Analyzer(config, use_cache=False).analyze(p, sample_rate=1e6)
    truth_payload = bytes.fromhex("".join(gt.payloads))
    assert res.payload.data and res.payload.data in truth_payload
    assert len(res.payload.data) >= len(truth_payload) // 2
