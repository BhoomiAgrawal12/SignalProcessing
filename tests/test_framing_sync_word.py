"""S10 sync-word extraction (report §5.2).

The constant-column run was rounded down to a byte boundary but its
trailing constant bytes were never removed, so a 24-bit run turned eb90
into eb9000.  It appeared in all ten operator reports.
"""
import numpy as np
import pytest

from rfanalyzer.framing.frames import (_trim_sync_word, analyze_frames,
                                       column_entropy, extract_sync_word)


def _frames(sync: bytes, filler: bytes, n=60, payload_len=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        body = sync + filler + bytes([i & 0xFF]) + bytes(
            rng.integers(0, 256, payload_len, dtype=np.uint8))
        rows.append(np.unpackbits(np.frombuffer(body, dtype=np.uint8)))
    return np.array(rows, dtype=np.uint8)


def test_trailing_zero_byte_is_not_part_of_the_sync_word():
    fr = _frames(b"\xeb\x90", b"\x00")
    got = extract_sync_word(fr, column_entropy(fr))
    assert got["found"]
    assert got["hex"] == "eb90"
    assert got["constant_run_bits"] > got["word_bits"]
    assert "trailing" in got["trimmed"]


def test_trailing_all_ones_byte_is_trimmed_too():
    fr = _frames(b"\x2d\xd4", b"\xff")
    got = extract_sync_word(fr, column_entropy(fr))
    assert got["hex"] == "2dd4"


def test_a_word_with_no_constant_tail_is_left_alone():
    fr = _frames(b"\x1a\xcf\xfc\x1d", b"")
    got = extract_sync_word(fr, column_entropy(fr))
    assert got["hex"] == "1acffc1d"
    assert "trimmed" not in got


def test_the_signature_library_overrides_the_trimming_rule():
    """A library match is direct evidence; the trailing-byte rule is an
    inference, so the library wins."""
    fr = _frames(b"\xeb\x90", b"\x00")
    got = extract_sync_word(fr, column_entropy(fr),
                            signature_words=["eb9000"])
    assert got["hex"] == "eb9000"
    assert "trimmed" not in got


def test_trimming_never_goes_below_two_bytes():
    """A one-byte sync word is not evidence of anything."""
    word = np.zeros(32, dtype=np.uint8)
    word[:16] = np.unpackbits(np.frombuffer(b"\xeb\x90", dtype=np.uint8))
    trimmed, _reason = _trim_sync_word(word)
    assert len(trimmed) >= 16


def test_payload_boundary_still_uses_the_whole_constant_run():
    """Trimming changes what is REPORTED as the sync word, not where the
    payload starts: a constant byte after the sync word is still not
    payload."""
    fr = _frames(b"\xeb\x90", b"\x00")
    analysis = analyze_frames(fr.reshape(-1), fr.shape[1])
    sync = analysis["sync"]
    assert sync["length_bits"] >= 24          # the full run
    assert sync["word_bits"] == 16            # the reported word
