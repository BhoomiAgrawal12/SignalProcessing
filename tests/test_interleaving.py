import numpy as np

from rfanalyzer.fec.conv import ConvCode, conv_encode
from rfanalyzer.interleaving import (block_interleave, block_deinterleave,
                                     helical_interleave, helical_deinterleave,
                                     conv_interleave, conv_deinterleave,
                                     pn_interleave, pn_deinterleave)
from rfanalyzer.interleaving.detect import identify_interleaver


def _coded(rng, n=8000):
    return conv_encode(rng.integers(0, 2, n).astype(np.uint8),
                       ConvCode(7, (0o171, 0o133)), terminate=False)


def test_roundtrips(rng):
    b = rng.integers(0, 2, 8000).astype(np.uint8)
    assert np.array_equal(block_deinterleave(block_interleave(b, 8, 16), 8, 16),
                          b[:7936])
    assert np.array_equal(
        helical_deinterleave(helical_interleave(b.copy(), 8, 16, 3), 8, 16, 3),
        b[:7936])
    assert np.array_equal(conv_deinterleave(conv_interleave(b, 4, 5), 4, 5), b)
    perm = rng.permutation(64)
    assert np.array_equal(pn_deinterleave(pn_interleave(b, perm), perm),
                          b[:len(b) // 64 * 64])


def test_identify_block(rng):
    inter = block_interleave(_coded(rng), 8, 16)
    top = identify_interleaver(inter, max_L=256)[0]
    assert top.kind == "block"
    assert (top.parameters["rows"], top.parameters["cols"]) == (8, 16)


def test_identify_block_with_offset(rng):
    inter = block_interleave(_coded(rng), 8, 16)[37:]
    top = identify_interleaver(inter, max_L=256)[0]
    assert top.kind == "block"
    assert (top.parameters["rows"], top.parameters["cols"]) == (8, 16)
    de = block_deinterleave(inter[top.parameters["offset"]:], 8, 16)
    from rfanalyzer.fec.detect import identify_convolutional
    hits = identify_convolutional(de)
    assert hits and hits[0]["syndrome_zero_rate"] > 0.99


def test_identify_none_on_plain_code(rng):
    top = identify_interleaver(_coded(rng), max_L=256)[0]
    assert top.kind == "none"


def test_identify_none_on_random(rng):
    bits = rng.integers(0, 2, 60000).astype(np.uint8)
    top = identify_interleaver(bits, max_L=256)[0]
    assert top.kind == "none"
