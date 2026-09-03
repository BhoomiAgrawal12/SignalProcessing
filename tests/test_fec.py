import numpy as np
import pytest

from rfanalyzer.fec.conv import ConvCode, conv_encode, viterbi_decode
from rfanalyzer.fec.rs import RSCode, symbols_to_bits
from rfanalyzer.fec.detect import identify_fec, identify_convolutional


def test_viterbi_clean(rng):
    code = ConvCode(7, (0o171, 0o133))
    msg = rng.integers(0, 2, 2000).astype(np.uint8)
    dec = viterbi_decode(conv_encode(msg, code), code)
    assert np.array_equal(dec[:len(msg)], msg)


def test_viterbi_noisy(rng):
    code = ConvCode(7, (0o171, 0o133))
    msg = rng.integers(0, 2, 4000).astype(np.uint8)
    enc = conv_encode(msg, code)
    enc[rng.random(len(enc)) < 0.05] ^= 1
    ber = (viterbi_decode(enc, code)[:len(msg)] != msg).mean()
    assert ber < 0.01


def test_rs_roundtrip(rng):
    rs = RSCode(255, 223)
    data = bytes(rng.integers(0, 256, 223, dtype=np.uint8))
    cw = bytearray(rs.encode(data))
    cw[5] ^= 0x55
    decoded, _ = rs.decode(bytes(cw))
    assert decoded[:223] == data


def test_identify_conv(config, rng):
    code = ConvCode(7, (0o171, 0o133))
    msg = rng.integers(0, 2, 8000).astype(np.uint8)
    enc = conv_encode(msg, code, terminate=False)
    hyp = identify_fec(enc, config.fec)[0]
    assert hyp.family == "convolutional"
    assert hyp.parameters["K"] == 7
    assert hyp.syndrome_zero_rate > 0.99
    assert np.array_equal(hyp.decoded_bits[:len(msg)], msg)


def test_identify_conv_inverted_substream(config, rng):
    """Rotation ambiguities invert substreams; the constancy test must
    still identify the code and resolve the inversion pattern."""
    code = ConvCode(7, (0o171, 0o133))
    msg = rng.integers(0, 2, 6000).astype(np.uint8)
    enc = conv_encode(msg, code, terminate=False)
    enc[0::2] ^= 1
    hyp = identify_fec(enc, config.fec)[0]
    assert hyp.family == "convolutional"
    assert np.array_equal(hyp.decoded_bits[:len(msg)], msg)


def test_identify_rs(config, rng):
    rs = RSCode(255, 223)
    stream = b"".join(rs.encode(bytes(rng.integers(0, 256, 223,
                                                   dtype=np.uint8)))
                      for _ in range(25))
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))
    hyp = identify_fec(bits.astype(np.uint8), config.fec)[0]
    assert hyp.family == "reed_solomon"
    assert hyp.parameters["n"] == 255 and hyp.parameters["k"] == 223


def test_identify_none_on_random(config, rng):
    bits = rng.integers(0, 2, 20000).astype(np.uint8)
    assert identify_fec(bits, config.fec)[0].family == "none"
