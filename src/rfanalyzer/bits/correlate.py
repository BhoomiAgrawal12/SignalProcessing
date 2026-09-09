"""Known-pattern correlation over a recovered bit stream.

The analyst workflow this serves: "I know this protocol starts with
A6 3C 91 - where is it, how good is the match, and what lies between the
occurrences?"  Answering it by hand means writing a one-off script per
recording, and the answer is only as good as the script.

Three things make this more than a substring search:

* The physical layer leaves ambiguities the analyst should not have to
  think about.  A BPSK stream can be inverted, a differentially-encoded
  one can be read either way, and byte order is a convention.  Each
  transform is searched and the one that matched is reported, so the
  result says WHICH reading of the stream contained the pattern.
* A real bit stream has errors, so the match must be approximate.  The
  score is the fraction of matching bits, and every reported hit carries
  the probability that a random stream would produce a match at least
  that good at at least one offset - the number that decides whether a
  hit means anything.
* Repeated hits at a regular spacing are a frame period, and the region
  between one hit's end and the next hit's start is the payload.  That
  is the actual deliverable.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def parse_pattern(text) -> np.ndarray:
    """Accept a pattern as hex bytes, a 0/1 bit string, or raw bytes.

    ``"A6 3C 91"``, ``"a63c91"``, ``"0xA63C91"`` and ``b"\\xa6<\\x91"`` all
    give the same 24 bits, MSB first.  A string of only 0s and 1s that is
    not a valid even-length hex string is read as bits, so ``"1011"`` is
    four bits rather than two nibbles.
    """
    if isinstance(text, np.ndarray):
        bits = np.asarray(text, dtype=np.uint8).ravel()
        if bits.size and bits.max() > 1:
            raise ValueError("bit array must contain only 0 and 1")
        return bits
    if isinstance(text, (bytes, bytearray)):
        return np.unpackbits(np.frombuffer(bytes(text), dtype=np.uint8))
    s = str(text).strip()
    if s.lower().startswith("0b"):
        s = s[2:]
        if not re.fullmatch(r"[01]+", s):
            raise ValueError(f"not a bit string: {text!r}")
        return np.array([int(c) for c in s], dtype=np.uint8)
    if s.lower().startswith("0x"):
        s = s[2:]
    cleaned = re.sub(r"[\s_:,-]", "", s)
    if re.fullmatch(r"[01]{2,}", cleaned) and len(cleaned) % 2:
        # odd length and only 0/1: unambiguously a bit string
        return np.array([int(c) for c in cleaned], dtype=np.uint8)
    if re.fullmatch(r"[0-9a-fA-F]+", cleaned) and len(cleaned) % 2 == 0:
        return np.unpackbits(np.frombuffer(bytes.fromhex(cleaned),
                                           dtype=np.uint8))
    if re.fullmatch(r"[01]+", cleaned):
        return np.array([int(c) for c in cleaned], dtype=np.uint8)
    raise ValueError(f"cannot parse {text!r} as hex bytes or a bit string")


def _poisson_tail(k: int, lam: float) -> float:
    """P(at least k events) for a Poisson mean ``lam``.

    Counting chance matches over many alignments is a rare-event count,
    so the Poisson tail is the right null: it answers "how surprising is
    it to see THIS MANY matches", which is the question a repeating
    preamble actually poses.
    """
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    if lam > 30.0:
        # exp(-lam) underflows below about lam = 745 and the term
        # recursion loses all precision long before that, so switch to
        # the normal approximation with a continuity correction, which is
        # accurate to better than 1% for lam above ~30
        z = (k - 0.5 - lam) / math.sqrt(lam)
        return float(0.5 * math.erfc(z / math.sqrt(2.0)))
    # P(X >= k) = 1 - sum_{i<k} e^-lam lam^i / i!
    term = math.exp(-lam)
    cdf = term
    for i in range(1, k):
        term *= lam / i
        cdf += term
        if cdf >= 1.0:
            return 0.0
    return float(max(0.0, 1.0 - cdf))


def _binomial_tail(n: int, k: int, p: float = 0.5) -> float:
    """P(at least k successes in n Bernoulli(p) trials)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if total >= 1.0:
            return 1.0
    return float(total)


def _match_counts(bits: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Number of matching bits for every alignment, via FFT correlation.

    Mapping both sequences to +-1 turns "count of agreements" into a
    correlation: agreements contribute +1 and disagreements -1, so
    matches = (n + correlation) / 2.  Doing it by FFT keeps a
    million-bit stream against a long preamble affordable, which matters
    because the search runs over every ambiguity transform.
    """
    n = len(pattern)
    if len(bits) < n or n == 0:
        return np.zeros(0, dtype=np.int64)
    a = np.asarray(bits, dtype=np.float64) * 2.0 - 1.0
    b = np.asarray(pattern, dtype=np.float64) * 2.0 - 1.0
    size = 1 << int(np.ceil(np.log2(len(a) + n)))
    corr = np.fft.irfft(np.fft.rfft(a, size) *
                        np.conj(np.fft.rfft(b, size)), size)
    corr = corr[: len(a) - n + 1]
    return np.rint((n + corr) / 2.0).astype(np.int64)


@dataclass
class PatternHit:
    offset_bits: int
    matching_bits: int
    pattern_bits: int
    transform: str
    p_value: float                      # chance of a match this good
    byte_aligned: bool = False

    @property
    def score(self) -> float:
        return self.matching_bits / max(1, self.pattern_bits)

    def to_dict(self) -> dict:
        return {"offset_bits": int(self.offset_bits),
                "offset_bytes": (self.offset_bits // 8
                                 if self.byte_aligned else None),
                "matching_bits": int(self.matching_bits),
                "pattern_bits": int(self.pattern_bits),
                "score": round(self.score, 4),
                "transform": self.transform,
                "byte_aligned": bool(self.byte_aligned),
                "p_value": self.p_value}


@dataclass
class CorrelationResult:
    pattern_bits: int
    pattern_hex: str
    stream_name: str = ""
    n_bits: int = 0
    hits: list = field(default_factory=list)
    period_bits: Optional[int] = None
    period_stability: Optional[float] = None
    payload_ranges: list = field(default_factory=list)
    best_transform: Optional[str] = None
    threshold_score: float = 0.0
    expected_by_chance: float = 0.0     # hits a random stream would give
    ensemble_p_value: float = 1.0       # P(this many hits | chance alone)
    significant: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {"pattern_hex": self.pattern_hex,
                "pattern_bits": self.pattern_bits,
                "stream": self.stream_name,
                "n_bits": self.n_bits,
                "threshold_score": round(self.threshold_score, 4),
                "n_hits": len(self.hits),
                "expected_by_chance": round(self.expected_by_chance, 4),
                "ensemble_p_value": self.ensemble_p_value,
                "significant": bool(self.significant),
                "hits": [h.to_dict() for h in self.hits],
                "best_transform": self.best_transform,
                "period_bits": self.period_bits,
                "period_stability": (None if self.period_stability is None
                                     else round(self.period_stability, 4)),
                "payload_ranges": self.payload_ranges,
                "note": self.note}


# Ambiguity transforms of a recovered bit stream.  These are the ones the
# physical layer genuinely leaves behind, so searching them is not
# fishing: it is undoing conventions the receiver could not resolve.
def _transforms(bits: np.ndarray, include_differential: bool = True) -> list:
    out = [("as recovered", bits),
           ("inverted", (1 - bits).astype(np.uint8))]
    if include_differential and len(bits) > 1:
        diff = np.concatenate(
            [[bits[0]], np.bitwise_xor(bits[1:], bits[:-1])]).astype(np.uint8)
        out.append(("differentially decoded", diff))
        out.append(("differentially decoded, inverted",
                    (1 - diff).astype(np.uint8)))
    return out


def correlate_pattern(bits, pattern, stream_name: str = "",
                      min_score: float = 0.85,
                      max_p_value: float = 1e-3,
                      max_hits: int = 64,
                      include_differential: bool = True,
                      include_reversed: bool = True) -> CorrelationResult:
    """Find ``pattern`` in ``bits`` under every physical-layer ambiguity.

    ``min_score`` is the fraction of bits that must match.  The hits are
    always reported; what is judged is whether they MEAN anything, and
    that is judged two ways because a preamble poses two different
    questions:

    * a single isolated occurrence has to be individually unlikely -
      ``n_alignments * P(>= k of n bits match)`` below ``max_p_value``;
    * a repeating preamble does not.  Sixteen bits recurring sixty times
      in a 46,000-bit stream is overwhelming evidence even though any one
      of those matches is unremarkable, so the ensemble is tested
      against a Poisson null on the number of hits.

    Judging only the first question is what makes a tool refuse to find
    a preamble that is plainly there.  Judging only the second lets a
    single lucky alignment look like a discovery.  Both are reported:
    ``significant`` is true when either test passes, and
    ``expected_by_chance`` says what a random stream would have given.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    pat = parse_pattern(pattern)
    pat_hex = np.packbits(
        np.concatenate([pat, np.zeros((-len(pat)) % 8, dtype=np.uint8)])
    ).tobytes().hex()
    res = CorrelationResult(pattern_bits=int(len(pat)), pattern_hex=pat_hex,
                            stream_name=stream_name, n_bits=int(len(bits)))
    if len(pat) == 0 or len(bits) < len(pat):
        res.note = "stream shorter than the pattern"
        return res

    n = len(pat)
    need = int(np.ceil(min_score * n))
    n_align = max(1, len(bits) - n + 1)
    res.threshold_score = need / n
    # what a random stream of this length would produce at this threshold,
    # counting every transform searched: each one is another family of
    # alignments and must be paid for
    n_variants = (4 if include_differential else 2) + \
        (1 if include_reversed else 0)
    res.expected_by_chance = float(
        n_variants * n_align * _binomial_tail(n, need))

    variants = _transforms(bits, include_differential)
    if include_reversed:
        variants.append(("bit-reversed pattern", None))
    all_hits = []
    for name, stream in variants:
        if stream is None:
            counts = _match_counts(bits, pat[::-1])
            name_used, target = name, pat[::-1]
        else:
            counts = _match_counts(stream, pat)
            name_used, target = name, pat
        if counts.size == 0:
            continue
        idx = np.nonzero(counts >= need)[0]
        for off in idx:
            k = int(counts[off])
            all_hits.append(PatternHit(
                offset_bits=int(off), matching_bits=k, pattern_bits=n,
                transform=name_used,
                p_value=float(min(1.0, n_align * _binomial_tail(n, k))),
                byte_aligned=bool(off % 8 == 0)))
    if not all_hits:
        res.note = (f"no alignment reaches {need}/{n} matching bits "
                    f"({res.threshold_score:.0%}); chance alone would "
                    f"have given about {res.expected_by_chance:.2g}")
        return res

    # keep the transform that produced the strongest evidence, then that
    # transform's hits in stream order: mixing transforms would make the
    # spacing between hits meaningless
    by_transform = {}
    for h in all_hits:
        by_transform.setdefault(h.transform, []).append(h)
    best_name = max(by_transform,
                    key=lambda t: (max(h.matching_bits
                                       for h in by_transform[t]),
                                   len(by_transform[t])))
    hits = sorted(by_transform[best_name], key=lambda h: h.offset_bits)
    res.best_transform = best_name
    res.hits = hits[:max_hits]

    # Is this evidence?  Either one hit is individually unlikely, or there
    # are far more of them than chance predicts.
    per_hit_p = min(h.p_value for h in hits)
    lam = res.expected_by_chance / max(1, n_variants)   # this transform's share
    res.ensemble_p_value = _poisson_tail(len(hits), lam)
    res.significant = bool(per_hit_p < max_p_value or
                           res.ensemble_p_value < max_p_value)

    if len(hits) >= 2:
        gaps = np.diff([h.offset_bits for h in hits])
        period = int(np.median(gaps))
        res.period_bits = period
        # how regular the spacing is: 1.0 means every gap is the period
        res.period_stability = float(
            np.mean(np.abs(gaps - period) <= max(1, period // 100)))
        for a, b in zip(hits, hits[1:]):
            start = a.offset_bits + n
            end = b.offset_bits
            if end > start:
                res.payload_ranges.append({
                    "start_bit": int(start), "end_bit": int(end),
                    "length_bits": int(end - start),
                    "byte_aligned": bool(start % 8 == 0 and
                                         (end - start) % 8 == 0)})
        res.note = (f"{len(hits)} occurrences at a median spacing of "
                    f"{period} bits ({res.period_stability:.0%} of gaps "
                    f"at that period); chance alone would have given about "
                    f"{lam:.2g}. The region between consecutive "
                    "occurrences is the payload.")
        # a regular spacing is evidence in its own right: chance matches
        # land where they like, they do not queue up on a grid
        if len(hits) >= 3 and (res.period_stability or 0) > 0.8:
            res.significant = True
    else:
        res.note = ("a single occurrence: no frame period can be inferred "
                    "from one hit"
                    + ("" if res.significant else
                       f"; chance alone would have given about {lam:.2g} "
                       "matches at this threshold, so this one is not "
                       "evidence on its own"))
    return res


def correlate_streams(streams: dict, pattern, **kw) -> dict:
    """Run :func:`correlate_pattern` over several named bit streams.

    The point of searching more than one is provenance: finding the
    preamble in the post-FEC stream and not in the raw one says the FEC
    hypothesis was right, and that is a stronger statement than either
    result alone.
    """
    out = {}
    for name, bits in streams.items():
        if bits is None or not len(bits):
            continue
        out[name] = correlate_pattern(bits, pattern, stream_name=name, **kw)
    return out


def best_result(results: dict):
    """The stream whose hits carry the strongest evidence."""
    scored = [r for r in results.values() if r.hits and r.significant]
    if not scored:
        return None
    return min(scored, key=lambda r: (r.ensemble_p_value,
                                      min(h.p_value for h in r.hits),
                                      -len(r.hits)))
