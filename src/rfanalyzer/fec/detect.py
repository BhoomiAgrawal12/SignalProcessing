"""Blind FEC identification (stage S9) - core IP.

Convolutional: the dual-code test.  For a rate-1/2 code with generators
(g1, g2), every coded stream satisfies g2(D)v1(D) + g1(D)v2(D) = 0.  We
sweep candidate (K, g1, g2) and both stream pairings; the true parameters
give a syndrome-zero rate near 1 while wrong ones sit near 0.5.

Reed-Solomon: sweep the candidate table over bit and symbol alignment,
scoring by all-zero-syndrome codeword fraction (cheap first pass on a few
codewords, confirmation pass on many).

The returned hypotheses carry `syndrome_zero_rate` - the objective
confidence metric that drives the retry loop.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..common.models import FECHypothesis
from .conv import ConvCode, STANDARD_CODES, viterbi_decode, conv_encode
from .rs import RSCode, bits_to_symbols

VERSION = 2


def _conv_syndrome_rate(bits: np.ndarray, g1: int, g2: int, K: int,
                        offset: int) -> float:
    """Fraction of zero syndromes for the dual-code parity check."""
    v = bits[offset:]
    n = (len(v) // 2) * 2
    v1 = v[0:n:2].astype(np.uint8)
    v2 = v[1:n:2].astype(np.uint8)
    m = len(v1) - K
    if m < 64:
        return 0.0
    syn = np.zeros(m, dtype=np.uint8)
    # bit i of g corresponds to delay K-1-i (MSB = delay 0), so indexing
    # v[i:i+m] with bit i implements the true parity convolution
    # g2(D) v1(D) + g1(D) v2(D) = 0 in encoder (MSB-first) convention.
    for i in range(K):
        if (g2 >> i) & 1:
            syn ^= v1[i:i + m]
        if (g1 >> i) & 1:
            syn ^= v2[i:i + m]
    return float(1.0 - syn.mean())


def _conv_syndrome_rate_signed(bits, g1, g2, K, offset):
    """Constancy test: substream inversions (produced by constellation
    rotation ambiguities, which are affine maps on the bit pairs) turn the
    dual-code syndrome into a *constant* 1 instead of 0.  Either constant
    value is therefore a detection; which substreams are inverted is
    resolved later at decode time. Returns (rate, syndrome_is_one)."""
    r = _conv_syndrome_rate(bits, g1, g2, K, offset)
    if (1.0 - r) > r:
        return 1.0 - r, True
    return r, False


def identify_convolutional(bits: np.ndarray, constraint_lengths=(3, 5, 7, 9),
                           exhaustive_max_k: int = 7,
                           max_test_bits: int = 100000,
                           min_rate: float = 0.7) -> list:
    """Sweep candidate rate-1/2 codes; returns hypotheses sorted by
    syndrome-zero rate."""
    bits = np.asarray(bits[:max_test_bits], dtype=np.uint8)
    results = []
    tested = set()

    def test(K, g1, g2):
        key = (K, g1, g2)
        if key in tested:
            return
        tested.add(key)
        for off in (0, 1):
            rate, syn_one = _conv_syndrome_rate_signed(bits, g1, g2, K, off)
            if rate >= min_rate:
                results.append({"K": K, "g1": g1, "g2": g2, "offset": off,
                                "syndrome_is_one": syn_one,
                                "syndrome_zero_rate": rate})

    # standard codes first (cheap, most likely)
    for code in STANDARD_CODES:
        g1, g2 = code.generators
        test(code.K, g1, g2)
    # exhaustive small-K sweep: both polys must tap first and last position
    for K in constraint_lengths:
        if K > exhaustive_max_k:
            continue
        top = 1 << (K - 1)
        cands = [top | 1 | (m << 1) for m in range(1 << (K - 2))]
        for g1, g2 in itertools.combinations(cands, 2):
            test(K, g1, g2)
            test(K, g2, g1)
    results.sort(key=lambda r: -r["syndrome_zero_rate"])
    return results


def identify_rs(bits: np.ndarray, candidates: list,
                quick_codewords: int = 8, confirm_codewords: int = 64) -> list:
    """Sweep RS candidates over bit offset (0..7) and symbol offset."""
    bits = np.asarray(bits, dtype=np.uint8)
    hits = []
    for cand in candidates:
        rs = RSCode(cand["n"], cand["k"], prim=cand.get("prim", 0x11D),
                    fcr=cand.get("fcr", 1), generator=cand.get("generator", 2))
        m = rs.m
        best = None
        for bit_off in range(m):
            symbols = bits_to_symbols(bits[bit_off:], m)
            # sweep codeword alignment; short captures may hold only a few
            # codewords, so test with whatever is available (>= 2)
            max_sym_off = min(rs.n, max(0, len(symbols) - 2 * rs.n) + 1)
            for sym_off in range(0, max_sym_off):
                seg = symbols[sym_off: sym_off + rs.n * quick_codewords]
                n_cw = len(seg) // rs.n
                if n_cw < 2:
                    break
                cw = seg[: n_cw * rs.n].reshape(n_cw, rs.n)
                syn = rs.syndromes(cw)
                rate = float((syn == 0).all(axis=1).mean())
                if rate > 0.5 and (best is None or rate > best["rate"]):
                    best = {"bit_offset": bit_off, "symbol_offset": sym_off,
                            "rate": rate}
        if best:
            # confirmation pass over many codewords
            symbols = bits_to_symbols(bits[best["bit_offset"]:], m)
            seg = symbols[best["symbol_offset"]:]
            n_cw = min(len(seg) // rs.n, confirm_codewords)
            cw = seg[: n_cw * rs.n].reshape(n_cw, rs.n)
            syn = rs.syndromes(cw)
            rate = float((syn == 0).all(axis=1).mean())
            hits.append({**cand, **best, "syndrome_zero_rate": rate,
                         "n_codewords_tested": n_cw})
    hits.sort(key=lambda h: -h["syndrome_zero_rate"])
    return hits


def identify_fec(bits: np.ndarray, config, llrs: np.ndarray = None,
                 try_rs: bool = True) -> list:
    """Full FEC identification: returns ranked FECHypothesis list, always
    ending with an explicit 'none' hypothesis."""
    bits = np.asarray(bits, dtype=np.uint8)
    hyps = []

    conv_hits = identify_convolutional(
        bits, tuple(config.conv_constraint_lengths),
        config.conv_exhaustive_max_k, config.max_test_bits,
        min_rate=config.min_syndrome_zero_rate)
    for hit in conv_hits[:3]:
        code = ConvCode(hit["K"], (hit["g1"], hit["g2"]))
        raw = bits[hit["offset"]:]
        raw = raw[: 2 * (len(raw) // 2)]
        # resolve the substream inversion pattern: syndrome==0 admits
        # (none, both), syndrome==1 admits (v1 only, v2 only); pick the
        # pattern whose decode re-encodes onto the observed stream
        patterns = ((0, 0), (1, 1)) if not hit.get("syndrome_is_one") \
            else ((1, 0), (0, 1))
        decoded, pre_ber, chosen = None, 1.0, None
        for inv1, inv2 in patterns:
            cand = raw.copy()
            if inv1:
                cand[0::2] ^= 1
            if inv2:
                cand[1::2] ^= 1
            dec = viterbi_decode(cand, code)
            re_enc = conv_encode(dec, code, terminate=False)
            L = min(len(re_enc), len(cand))
            ber = float((re_enc[:L] != cand[:L]).mean())
            if ber < pre_ber:
                decoded, pre_ber, chosen = dec, ber, (inv1, inv2)
            if ber < 0.02:
                break
        if decoded is None or pre_ber > 0.2:
            # decode inconsistent with observed stream: reject (protects
            # against equivalent-but-wrong trellis descriptions)
            continue
        hyps.append(FECHypothesis(
            family="convolutional", code_rate=0.5,
            parameters={"K": hit["K"], "g1_octal": oct(hit["g1"]),
                        "g2_octal": oct(hit["g2"]), "rate": "1/2",
                        "stream_offset": hit["offset"],
                        "substream_inversion": list(chosen)},
            syndrome_zero_rate=hit["syndrome_zero_rate"],
            decoded_bits=decoded, pre_fec_ber=pre_ber,
            score=hit["syndrome_zero_rate"] * (1.0 - min(1.0, 2 * pre_ber))))

    # The RS alignment sweep is the expensive part of this stage; a
    # confident convolutional hit makes it redundant (concatenated codes
    # are handled by re-running identification on the decoded stream).
    best_conv = conv_hits[0]["syndrome_zero_rate"] if conv_hits else 0.0
    rs_hits = [] if (best_conv > 0.9 or not try_rs) else \
        identify_rs(bits, config.rs_candidates)
    for hit in rs_hits[:2]:
        if hit["syndrome_zero_rate"] < config.min_syndrome_zero_rate * 0.5:
            continue
        rs = RSCode(hit["n"], hit["k"], prim=hit.get("prim", 0x11D),
                    fcr=hit.get("fcr", 1), generator=hit.get("generator", 2))
        decoded = _rs_decode_stream(bits, rs, hit["bit_offset"],
                                    hit["symbol_offset"])
        hyps.append(FECHypothesis(
            family="reed_solomon", code_rate=hit["k"] / hit["n"],
            parameters={"n": hit["n"], "k": hit["k"], "fcr": hit.get("fcr"),
                        "prim": hex(hit.get("prim", 0x11D)),
                        "bit_offset": hit["bit_offset"],
                        "symbol_offset": hit["symbol_offset"],
                        "n_codewords_tested": hit["n_codewords_tested"]},
            syndrome_zero_rate=hit["syndrome_zero_rate"],
            decoded_bits=decoded,
            score=hit["syndrome_zero_rate"] * 0.98))

    hyps.append(FECHypothesis(family="none", score=0.2,
                              parameters={"reason": "no code identified above threshold"
                                          if not hyps else "fallback"}))
    hyps.sort(key=lambda h: -h.score)
    return hyps


def _rs_decode_stream(bits: np.ndarray, rs: RSCode, bit_off: int,
                      sym_off: int) -> np.ndarray:
    from .rs import symbols_to_bits
    symbols = bits_to_symbols(bits[bit_off:], rs.m)[sym_off:]
    n_cw = len(symbols) // rs.n
    out = []
    for i in range(n_cw):
        cw = bytes(symbols[i * rs.n:(i + 1) * rs.n].astype(np.uint8))
        try:
            data, _ = rs.decode(cw)
            out.append(np.frombuffer(data[:rs.k], dtype=np.uint8))
        except Exception:
            continue
    if not out:
        return np.zeros(0, dtype=np.uint8)
    data_syms = np.concatenate(out).astype(np.int64)
    return symbols_to_bits(data_syms, 8)
