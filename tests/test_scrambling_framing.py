import numpy as np

from rfanalyzer.scrambling.lfsr import LFSR, additive_scramble, KNOWN_WHITENERS
from rfanalyzer.scrambling.berlekamp import (berlekamp_massey, align_whitener)
from rfanalyzer.framing.crc import crc_compute, crc_hunt
from rfanalyzer.framing.frames import (find_frame_length, analyze_frames,
                                       payload_stats)
from rfanalyzer.fec.conv import ConvCode, conv_encode


def test_berlekamp_massey_recovers_lfsr():
    seq = LFSR(0x221, 0x1AB).sequence(600)
    poly, L = berlekamp_massey(seq[:200])
    assert L == 9
    from rfanalyzer.scrambling.berlekamp import bm_to_lfsr_poly
    lp = bm_to_lfsr_poly(poly, L)
    assert lp == 0x221
    regen = LFSR(lp, int(sum(int(b) << i for i, b in enumerate(seq[:9])))).sequence(600)
    assert (regen == seq).mean() > 0.99


def test_align_whitener(rng):
    coded = conv_encode(rng.integers(0, 2, 6000).astype(np.uint8),
                        ConvCode(7, (0o171, 0o133)), terminate=False)
    w = KNOWN_WHITENERS["PN9-CC1101"]
    scrambled = additive_scramble(coded, w["poly"], w["seed"])[301:]
    res = align_whitener(scrambled, "PN9-CC1101", P=32)
    assert res["phase"] == 301 % 511
    from rfanalyzer.fec.detect import identify_convolutional
    hits = identify_convolutional(res["bits"])
    assert hits and hits[0]["syndrome_zero_rate"] > 0.99


def test_crc_compute_known_vector():
    # CRC-16/CCITT-FALSE of "123456789" is 0x29B1
    assert crc_compute(b"123456789", 16, 0x1021, 0xFFFF, False, False, 0) == 0x29B1
    # CRC-32 of "123456789" is 0xCBF43926
    assert crc_compute(b"123456789", 32, 0x04C11DB7, 0xFFFFFFFF, True, True,
                       0xFFFFFFFF) == 0xCBF43926


def _frames(rng, n=50):
    out = []
    for i in range(n):
        body = (0xEB90).to_bytes(2, "big") + bytes([i & 0xFF, 1]) + \
            bytes(rng.integers(32, 127, 6, dtype=np.uint8))
        crc = crc_compute(body, 16, 0x1021, 0xFFFF, False, False, 0)
        out.append(np.unpackbits(np.frombuffer(body + crc.to_bytes(2, "big"),
                                               dtype=np.uint8)))
    return np.concatenate(out).astype(np.uint8)


def test_frame_length_and_crc(rng):
    bits = _frames(rng)
    cands = find_frame_length(bits, 16, 512)
    assert cands[0]["length"] == 96
    fr = analyze_frames(bits, 96)
    assert fr["sync"]["found"] and fr["sync"]["hex"].startswith("eb90")
    hits = crc_hunt(fr["frames"])
    assert hits and hits[0]["name"] == "CRC-16-CCITT-FALSE"
    assert hits[0]["pass_fraction"] == 1.0


def test_payload_stats_encrypted_detection(rng):
    random_data = bytes(rng.integers(0, 256, 4096, dtype=np.uint8))
    assert payload_stats(random_data)["likely_encrypted"]
    text = b"the quick brown fox jumps over the lazy dog " * 80
    assert not payload_stats(text)["likely_encrypted"]
