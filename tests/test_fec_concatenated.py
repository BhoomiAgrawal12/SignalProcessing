"""Concatenated codes: an outer block code around an inner trellis code.

The outer code is invisible on the received stream - it only appears once
the inner decoder has run - so identifying it needs a second pass over
the DECODED bits rather than another sweep of the same stream.  This is
the shape of most satellite and deep-space telemetry, AO-73's published
format included (CCSDS r=1/2 K=7 inside an interleaved RS pair).
"""
import numpy as np
import pytest

from rfanalyzer.common.config import Config
from rfanalyzer.fec.conv import ConvCode, conv_encode
from rfanalyzer.fec.detect import identify_fec

reedsolo = pytest.importorskip("reedsolo",
                               reason="Reed-Solomon needs the reedsolo extra")
from rfanalyzer.fec.rs import RSCode          # noqa: E402  (after skip)


@pytest.fixture(scope="module")
def fec_config():
    return Config().fec


def _concatenated(n_blocks=10, seed=4):
    """data -> RS(255,223) -> convolutional r=1/2 K=7."""
    rng = np.random.default_rng(seed)
    rs = RSCode(255, 223, fcr=1, generator=2)
    data = bytes(rng.integers(0, 256, 223 * n_blocks, dtype=np.uint8))
    cws = b"".join(rs.encode(data[i:i + 223])
                   for i in range(0, len(data), 223))
    outer_bits = np.unpackbits(np.frombuffer(cws, dtype=np.uint8))
    coded = conv_encode(outer_bits, ConvCode(7, (0o171, 0o133)),
                        terminate=False)
    return coded, data, outer_bits


@pytest.mark.slow
def test_both_stages_are_identified(fec_config):
    coded, _data, _outer = _concatenated()
    top = identify_fec(coded, fec_config)[0]
    assert top.family == "concatenated"
    assert top.parameters["inner"]["family"] == "convolutional"
    assert top.parameters["inner"]["g1_octal"] == oct(0o171)
    assert top.parameters["outer"]["n"] == 255
    assert top.parameters["outer"]["k"] == 223


@pytest.mark.slow
def test_the_reported_rate_is_the_product_of_both_stages(fec_config):
    coded, _data, _outer = _concatenated()
    top = identify_fec(coded, fec_config)[0]
    assert top.code_rate == pytest.approx(0.5 * 223 / 255, rel=1e-6)


@pytest.mark.slow
def test_the_decoded_bits_are_the_original_data(fec_config):
    coded, data, _outer = _concatenated()
    top = identify_fec(coded, fec_config)[0]
    truth = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    got = np.asarray(top.decoded_bits, dtype=np.uint8)
    n = min(len(got), len(truth))
    assert n > 8192
    assert float((got[:n] == truth[:n]).mean()) == pytest.approx(1.0)


@pytest.mark.slow
def test_a_concatenated_hypothesis_must_beat_its_own_inner_stage(fec_config):
    """A spurious outer match must not displace a good simple answer."""
    rng = np.random.default_rng(2)
    plain = conv_encode(rng.integers(0, 2, 40000).astype(np.uint8),
                        ConvCode(7, (0o171, 0o133)), terminate=False)
    hyps = identify_fec(plain, fec_config)
    assert hyps[0].family == "convolutional"
    concat = [h for h in hyps if h.family == "concatenated"]
    assert all(h.score <= hyps[0].score for h in concat)


@pytest.mark.slow
def test_a_single_stage_stream_is_not_reported_as_concatenated(fec_config):
    rng = np.random.default_rng(6)
    rs = RSCode(255, 223, fcr=1, generator=2)
    data = bytes(rng.integers(0, 256, 223 * 8, dtype=np.uint8))
    cws = b"".join(rs.encode(data[i:i + 223])
                   for i in range(0, len(data), 223))
    bits = np.unpackbits(np.frombuffer(cws, dtype=np.uint8))
    top = identify_fec(bits, fec_config)[0]
    assert top.family == "reed_solomon"


@pytest.mark.slow
def test_random_data_yields_no_concatenated_claim(fec_config):
    rng = np.random.default_rng(8)
    noise = rng.integers(0, 2, 60000).astype(np.uint8)
    hyps = identify_fec(noise, fec_config)
    assert all(h.family != "concatenated" for h in hyps)
