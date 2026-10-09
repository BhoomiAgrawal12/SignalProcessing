import numpy as np

from dhwani.bits.packing import pack_bits, unpack_bits, pack_rows, popcount_u64
from dhwani.gf2.rank import gf2_rank_bits, rank_profile


def test_pack_roundtrip(rng):
    bits = rng.integers(0, 2, 1000).astype(np.uint8)
    assert np.array_equal(unpack_bits(pack_bits(bits), 1000), bits)
    assert popcount_u64(pack_bits(bits)).sum() == bits.sum()


def test_pack_rows(rng):
    bits = rng.integers(0, 2, 960).astype(np.uint8)
    M = pack_rows(bits, 60)
    for i in range(4):
        assert np.array_equal(unpack_bits(M[i], 60), bits[i * 60:(i + 1) * 60])


def test_rank_identity():
    I = np.eye(64, dtype=np.uint8).ravel()
    assert gf2_rank_bits(I, 64)[0] == 64


def test_rank_random_full(rng):
    bits = rng.integers(0, 2, 128 * 200).astype(np.uint8)
    assert gf2_rank_bits(bits, 128, 200)[0] == 128


def test_rank_codewords(rng):
    G = rng.integers(0, 2, (8, 16)).astype(np.uint8)
    G[:, :8] = np.eye(8)
    code = (rng.integers(0, 2, (400, 8)).astype(np.uint8) @ G) % 2
    assert gf2_rank_bits(code.ravel().astype(np.uint8), 16, 300)[0] == 8


def test_rank_profile_flat_on_random(rng):
    bits = rng.integers(0, 2, 40000).astype(np.uint8)
    prof = rank_profile(bits, 2, 128)
    assert prof["significance"].max() < 3.0
