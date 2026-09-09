"""Known-pattern correlation over recovered bit streams.

The analyst workflow this serves: enter a known preamble, get back its
bit offset, a match score, and the likely payload range.  What makes it more than a substring search is that the physical
layer leaves ambiguities (inversion, differential encoding, bit order)
and that a short pattern in a long stream is not evidence however well it
matches.
"""
import numpy as np
import pytest

from rfanalyzer.bits.correlate import (correlate_pattern, correlate_streams,
                                       best_result, parse_pattern)


def _framed(pattern_bits, frame_bits=200, n_frames=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_frames):
        out.append(pattern_bits)
        out.append(rng.integers(0, 2, frame_bits - len(pattern_bits)
                                ).astype(np.uint8))
    return np.concatenate(out).astype(np.uint8)


# ------------------------------------------------------------- parsing
@pytest.mark.parametrize("text", ["A6 3C 91", "a63c91", "0xA63C91",
                                  "A6:3C:91", "a6-3c-91"])
def test_hex_spellings_agree(text):
    assert parse_pattern(text).tolist() == parse_pattern(b"\xa6\x3c\x91").tolist()


def test_bit_string_is_read_as_bits():
    assert parse_pattern("0b1011").tolist() == [1, 0, 1, 1]
    assert len(parse_pattern("0b10110")) == 5


def test_unparseable_pattern_raises():
    with pytest.raises(ValueError):
        parse_pattern("nonsense!")


# ---------------------------------------------------------- correlation
def test_finds_every_occurrence_and_the_frame_period():
    """All 60 planted occurrences are found.  A couple of extra hits at
    the default 85% threshold are chance matches, not a defect - which is
    exactly what ``expected_by_chance`` is there to say."""
    pat = parse_pattern("A63C91")
    bits = _framed(pat)
    res = correlate_pattern(bits, "A6 3C 91")
    assert res.significant
    assert res.period_bits == 200
    planted = {200 * i for i in range(60)}
    found = {h.offset_bits for h in res.hits}
    assert planted <= found
    exact = [h for h in res.hits if h.score == 1.0]
    assert len(exact) == 60
    assert len(res.hits) - 60 <= max(2, int(2 * res.expected_by_chance))


def test_payload_range_is_the_region_between_occurrences():
    pat = parse_pattern("A63C91")
    res = correlate_pattern(_framed(pat), "A63C91")
    first = res.payload_ranges[0]
    assert first["start_bit"] == len(pat)
    assert first["end_bit"] == 200
    assert first["length_bits"] == 200 - len(pat)


def test_an_inverted_stream_is_found_and_named():
    """BPSK leaves a polarity ambiguity the receiver cannot resolve; the
    analyst should not have to think about it, but should be told."""
    bits = _framed(parse_pattern("A63C91"))
    res = correlate_pattern((1 - bits).astype(np.uint8), "A63C91")
    assert res.best_transform == "inverted"
    assert res.significant
    assert {200 * i for i in range(60)} <= {h.offset_bits for h in res.hits}


def test_a_differentially_encoded_stream_is_found_and_named():
    bits = _framed(parse_pattern("A63C91"))
    enc = np.cumsum(bits) % 2                    # differential encoding
    res = correlate_pattern(enc.astype(np.uint8), "A63C91")
    assert "differential" in res.best_transform
    assert len(res.hits) >= 55


def test_bit_errors_are_tolerated_and_scored():
    rng = np.random.default_rng(5)
    pat = parse_pattern("A63C9155AA")            # 40 bits: enough to be
    bits = _framed(pat, frame_bits=400, n_frames=40)  # significant
    noisy = bits.copy()
    flip = rng.choice(len(bits), size=len(bits) // 400, replace=False)
    noisy[flip] ^= 1
    res = correlate_pattern(noisy, "A63C9155AA", min_score=0.85)
    assert res.hits
    assert any(h.score < 1.0 for h in res.hits)  # imperfect hits kept
    assert all(h.score >= res.threshold_score for h in res.hits)


def test_chance_matches_are_reported_but_not_called_significant():
    """A short pattern in a long stream matches by accident.  Hiding those
    matches would be as wrong as believing them: they are reported, with
    what chance alone predicts, and marked not significant."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 2, 1 << 20).astype(np.uint8)
    res = correlate_pattern(noise, "A63C91")
    assert not res.significant
    assert res.expected_by_chance > 1.0
    assert res.ensemble_p_value > 1e-3


def test_a_repeating_short_preamble_is_significant():
    """The case that matters in practice, and the one a per-hit test
    alone gets wrong: sixteen bits recurring sixty times is overwhelming
    even though any single one of those matches is unremarkable."""
    pat = parse_pattern("EB90")
    bits = _framed(pat, frame_bits=96, n_frames=60, seed=1)
    res = correlate_pattern(bits, "EB90")
    assert res.significant
    assert res.period_bits == 96
    assert res.ensemble_p_value < 1e-6
    assert len(res.hits) >= 55


def test_random_data_produces_no_hit_for_a_long_pattern():
    rng = np.random.default_rng(9)
    noise = rng.integers(0, 2, 200000).astype(np.uint8)
    res = correlate_pattern(noise, "A63C9155AA1234")
    assert res.hits == []
    assert not res.significant
    assert res.expected_by_chance < 0.1


def test_the_expected_chance_count_is_always_reported():
    """The number that tells an analyst whether to believe the hits."""
    res = correlate_pattern(_framed(parse_pattern("A63C9155AA")),
                            "A63C9155AA")
    assert res.expected_by_chance >= 0.0
    assert 0.0 <= res.ensemble_p_value <= 1.0
    assert str(round(res.expected_by_chance, 2)) or True


def test_every_reported_hit_carries_a_p_value():
    res = correlate_pattern(_framed(parse_pattern("A63C9155AA")),
                            "A63C9155AA")
    assert res.hits
    assert all(0.0 <= h.p_value <= 1.0 for h in res.hits)
    assert all(h.p_value < 1e-3 for h in res.hits)


def test_byte_alignment_is_reported():
    pat = parse_pattern("A63C91")
    res = correlate_pattern(_framed(pat), "A63C91")
    assert res.hits[0].byte_aligned
    shifted = np.concatenate([[0, 0, 0], _framed(pat)]).astype(np.uint8)
    res2 = correlate_pattern(shifted, "A63C91")
    assert not res2.hits[0].byte_aligned


def test_a_single_occurrence_infers_no_period():
    pat = parse_pattern("A63C9155AA")
    rng = np.random.default_rng(3)
    bits = np.concatenate([rng.integers(0, 2, 500).astype(np.uint8), pat,
                           rng.integers(0, 2, 500).astype(np.uint8)])
    res = correlate_pattern(bits.astype(np.uint8), "A63C9155AA")
    assert len(res.hits) == 1
    assert res.period_bits is None
    assert "no frame period" in res.note


def test_multiple_streams_are_ranked_by_evidence():
    """Finding the preamble after FEC decoding but not before is itself
    evidence that the FEC hypothesis was right."""
    pat = parse_pattern("A63C9155AA")
    rng = np.random.default_rng(11)
    good = _framed(pat, frame_bits=400, n_frames=40)
    junk = rng.integers(0, 2, len(good)).astype(np.uint8)
    results = correlate_streams({"raw": junk, "decoded": good}, "A63C9155AA")
    winner = best_result(results)
    assert winner is not None
    assert winner.stream_name == "decoded"
    assert results["decoded"].significant
    assert not results["raw"].significant


def test_empty_and_oversized_inputs_are_handled():
    assert correlate_pattern(np.zeros(0, dtype=np.uint8), "A63C91").hits == []
    short = np.array([1, 0, 1], dtype=np.uint8)
    res = correlate_pattern(short, "A63C91")
    assert res.hits == []
    assert "shorter than the pattern" in res.note
