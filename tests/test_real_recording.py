"""The one real capture in the repository: AO-73 (FUNcube-1).

Report §11 ranks adding this first, ahead of any further synthetic case:
it is cheap, and it turns the largest known failure into something
measurable.  Every threshold here is checked against PUBLISHED ground
truth (the AO-40 FEC telemetry format designed by Phil Karn KA9Q), not
against what the tool happens to produce.
"""
import os

import numpy as np
import pytest

from rfanalyzer.channelization import channelize
from rfanalyzer.common.config import Config
from rfanalyzer.conditioning import condition
from rfanalyzer.demod import demodulate
from rfanalyzer.detection import detect_signals
from rfanalyzer.ingestion import load_recording
from rfanalyzer.modulation import classify_modulation
from rfanalyzer.params import estimate_parameters

AO73_PATH = os.path.join(os.path.dirname(__file__), "..", "SelfRun",
                         "ao73.wav")
# published ground truth
TRUE_SYMBOL_RATE_HZ = 1200.0
TRUE_CARRIER_HZ = 1102.0
TRUE_OCCUPIED_BW_HZ = 2000.0
TRUE_CONV_K = 7
TRUE_GENERATORS = {0o171, 0o133}
TRUE_INTERLEAVER = (80, 65)          # 80 x 65 bit block = 5200 bits

pytestmark = pytest.mark.skipif(not os.path.exists(AO73_PATH),
                                reason="ao73.wav not present")


@pytest.fixture(scope="module")
def front_end():
    cfg = Config()
    cfg.modulation.cvnet_enabled = False
    rec = load_recording(AO73_PATH)
    x, cond = condition(rec.samples, real_signal=rec.is_real_signal)
    segs, dbg = detect_signals(x, cfg.cfar, sample_rate=rec.sample_rate,
                               positive_only=rec.is_real_signal)
    assert segs, "no segment detected in the AO-73 capture"
    out = None
    for seg in segs[:3]:
        ch = channelize(x, seg)
        p = estimate_parameters(ch["samples"], cfg.params,
                                sample_rate=ch["sample_rate"],
                                analysis_band_norm=ch["analysis_band_norm"])
        m = classify_modulation(ch["samples"], p, cfg.modulation)
        out = {"cfg": cfg, "rec": rec, "cond": cond, "segments": segs,
               "detection": dbg, "segment": seg, "ch": ch, "params": p,
               "modulation": m}
        if m.prediction != "UNKNOWN":
            break
    return out


def test_mono_wav_is_recognised_as_real_and_made_analytic(front_end):
    assert front_end["rec"].is_real_signal
    assert front_end["cond"].analytic_conversion


def test_detection_isolates_the_carrier_instead_of_the_audio_band(front_end):
    """The shipped detector returned one segment spanning -8.2 to
    +8.2 kHz - the whole audio band - for a 2 kHz emission (§10.2)."""
    seg = front_end["segment"]
    sr = front_end["rec"].sample_rate
    bandwidth = seg.bandwidth_norm * sr
    assert bandwidth < 4 * TRUE_OCCUPIED_BW_HZ
    assert seg.f_low_norm * sr <= TRUE_CARRIER_HZ <= seg.f_high_norm * sr


def test_symbol_rate_matches_the_published_1200_baud(front_end):
    """It was 15,159 Bd - 12.6x high - and the true value was not even
    among the five ranked candidates (§10.2)."""
    p = front_end["params"]
    assert p.symbol_rate_hz is not None
    assert p.symbol_rate_hz == pytest.approx(TRUE_SYMBOL_RATE_HZ, rel=0.05)


def test_samples_per_symbol_is_in_the_audio_telemetry_range(front_end):
    """The pipeline settled on 3.166 sps for a signal at 40 sps in the
    original recording and never questioned it (§10.5 item 2)."""
    p = front_end["params"]
    assert p.samples_per_symbol > 8.0


def test_modulation_is_not_unknown(front_end):
    """`UNKNOWN` stopped the chain at S5 and the bit layer never ran."""
    assert front_end["modulation"].prediction != "UNKNOWN"


def test_the_carrier_cross_check_is_reported(front_end):
    """S4's own answer disagreed with the x^2 line by three orders of
    magnitude and nothing reconciled the two (§10.5 item 3)."""
    cross = front_end["detection"].get("carrier_cross_check")
    assert cross is not None and "carrier_norm" in cross


@pytest.mark.slow
def test_the_bit_layer_recovers_the_published_convolutional_code(front_end):
    """The claim §10.3 established by hand, now reached blind: the
    project's own FEC identifier finds the CCSDS r=1/2 k=7 code in a real
    satellite recording, once the front end hands it conditioned bits."""
    from rfanalyzer.fec.detect import identify_convolutional
    from rfanalyzer.interleaving.interleavers import block_deinterleave

    p, ch, cfg = front_end["params"], front_end["ch"], front_end["cfg"]
    res = demodulate(ch["samples"], "BPSK", p.samples_per_symbol or 8.0,
                     cfg.demod)
    assert res.hard_bits is not None and len(res.hard_bits) > 5200

    bits = np.asarray(res.hard_bits, dtype=np.uint8)
    # AO-73 is DBPSK, so the differential branch of the S7 fan-out is the
    # one that matters here
    diff = np.concatenate(
        [[bits[0]], np.bitwise_xor(bits[1:], bits[:-1])]).astype(np.uint8)
    n = (len(diff) // 5200) * 5200
    deinterleaved = block_deinterleave(diff[:n], *TRUE_INTERLEAVER)
    hits = identify_convolutional(deinterleaved, [3, 5, 7, 9])
    assert hits, "no convolutional code identified"
    best = hits[0]
    assert best["K"] == TRUE_CONV_K
    assert {best["g1"], best["g2"]} == TRUE_GENERATORS
    assert best["syndrome_zero_rate"] > 0.7


@pytest.mark.slow
def test_the_code_is_not_found_before_de_interleaving(front_end):
    """The other half of the §10.3 claim: only the true 80x65 geometry
    exposes the code, which is what makes the identification evidence
    rather than coincidence."""
    from rfanalyzer.fec.detect import identify_convolutional

    p, ch, cfg = front_end["params"], front_end["ch"], front_end["cfg"]
    res = demodulate(ch["samples"], "BPSK", p.samples_per_symbol or 8.0,
                     cfg.demod)
    bits = np.asarray(res.hard_bits, dtype=np.uint8)
    diff = np.concatenate(
        [[bits[0]], np.bitwise_xor(bits[1:], bits[:-1])]).astype(np.uint8)
    hits = identify_convolutional(diff[:(len(diff) // 5200) * 5200],
                                  [3, 5, 7, 9])
    interleaved_rate = hits[0]["syndrome_zero_rate"] if hits else 0.0
    assert interleaved_rate < 0.7
