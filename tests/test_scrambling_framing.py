import numpy as np
import pytest

from dhwani.scrambling.lfsr import LFSR, additive_scramble, KNOWN_WHITENERS
from dhwani.scrambling.berlekamp import (berlekamp_massey, align_whitener)
from dhwani.framing.crc import crc_compute, crc_hunt
from dhwani.framing.frames import (find_frame_length, analyze_frames,
                                       payload_stats)
from dhwani.fec.conv import ConvCode, conv_encode


def test_berlekamp_massey_recovers_lfsr():
    seq = LFSR(0x221, 0x1AB).sequence(600)
    poly, L = berlekamp_massey(seq[:200])
    assert L == 9
    from dhwani.scrambling.berlekamp import bm_to_lfsr_poly
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
    from dhwani.fec.detect import identify_convolutional
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


def test_truncated_whitener_search_is_reported():
    """The 512-phase DVB search ceiling is stated when it could matter:
    no scrambler found and no CRC validation."""
    from dhwani.common.models import AnalysisHypothesis, AnalysisResult
    from dhwani.pipeline import _fill_result_from_best
    best = {"candidate": {"whitener": {"name": "none"}}, "scrambler_bm": {},
            "interleaver": None, "fec": None, "frames": None,
            "crc_hits": [], "decoded_bits": None,
            "hypothesis": AnalysisHypothesis(stage="fec")}
    res = AnalysisResult(run_id="t")
    _fill_result_from_best(res, best)
    assert any("DVB" in w and "512" in w for w in res.warnings)
    best["crc_hits"] = [{"name": "CRC-16"}]
    res = AnalysisResult(run_id="t")
    _fill_result_from_best(res, best)
    assert not any("DVB" in w for w in res.warnings)


@pytest.mark.parametrize("skew", [1, -1])
@pytest.mark.parametrize("turn", [0, 1])
def test_oqpsk_stagger_pairing_is_in_the_fan_out(rng, skew, turn):
    """A blind OQPSK receiver can pair each I with the neighbouring Q, on
    either side; the bits are intact but misaligned (measured: BER 0.20 with
    GOOD EVM, 0 after re-pairing). A 90-degree lock swaps the rails, so both
    skews must be offered under every rotation."""
    from dhwani.bits.ambiguity import enumerate_ambiguities
    from dhwani.demod.constellations import bits_to_iq_symbols
    bits = rng.integers(0, 2, 2000).astype(np.uint8)
    s = bits_to_iq_symbols(bits, "OQPSK")
    mispaired = (s.real + 1j * np.roll(s.imag, skew)) * 1j ** turn
    streams = enumerate_ambiguities(mispaired, "OQPSK")
    # re-pairing may start one symbol later; framing absorbs that offset
    assert any(np.array_equal(st.bits[10:-10], bits[10 + d:len(bits) - 10 + d])
               for st in streams for d in (-2, 0, 2))


def test_whitener_phase_from_sync_repeats():
    """A1: the right phase makes the sync word repeat once per frame; a
    wrong whitener or random data stays near the random maximum."""
    from dhwani.scrambling.berlekamp import raw_word_repeats, whitener_by_repeats
    from dhwani.synth.factory import WaveformFactory
    bits, _, _ = WaveformFactory(seed=3).build_frames(80)
    w = KNOWN_WHITENERS["PN9-CC1101"]
    sc = additive_scramble(bits, w["poly"], w["seed"])[137:]
    assert raw_word_repeats(sc) < 8
    hit = whitener_by_repeats(sc, "PN9-CC1101")
    assert hit["phase"] == 137 and hit["repeats"] >= 70
    assert whitener_by_repeats(sc, "CCSDS")["repeats"] < 16


def _crc_reference(data, width, poly, init, refin, refout, xorout):
    """Bit-at-a-time reference, kept independent of the table version."""
    def refl(v, w):
        return int(f"{v:0{w}b}"[::-1], 2)
    crc, top, mask = init, 1 << (width - 1), (1 << width) - 1
    for byte in data:
        byte = refl(byte, 8) if refin else byte
        crc ^= byte << (width - 8)
        for _ in range(8):
            crc = (((crc << 1) ^ poly) if crc & top else (crc << 1)) & mask
    return (refl(crc, width) if refout else crc) ^ xorout


def test_table_crc_matches_bitwise_reference(rng):
    """A4: the table-driven CRC is bit-exact for every configured preset."""
    from dhwani.common.config import FramingConfig
    for cand in FramingConfig().crc_candidates:
        args = {k: cand[k] for k in ("width", "poly", "init", "refin",
                                     "refout", "xorout")}
        for n in (0, 1, 7, 33):
            data = bytes(rng.integers(0, 256, n, dtype=np.uint8))
            assert crc_compute(data, **args) == _crc_reference(data, **args)


def test_crc_hunt_keeps_exact_threshold_hits():
    """Early abort must not drop a candidate sitting exactly on the
    pass-fraction threshold (9 of 10 frames at 0.9)."""
    frames = []
    for i in range(10):
        body = bytes([0xEB, 0x90, i, 1, 2, 3])
        crc = crc_compute(body, 16, 0x1021, 0xFFFF, False, False, 0)
        if i == 4:
            crc ^= 1                                   # one corrupted frame
        frames.append(np.unpackbits(np.frombuffer(body + crc.to_bytes(2, "big"),
                                                  dtype=np.uint8)))
    hits = crc_hunt(np.array(frames))
    assert any(h["name"] == "CRC-16-CCITT-FALSE" and h["passes"] == 9
               for h in hits)
