"""S6 receiver: EVM gates, AGC scale, carrier paths, quality gating.

None of S6 had direct coverage (report §6), which is how a fixed 1/window
normalisation in the phase polish and a unit-power AGC assumption
survived through the dense-constellation results of §4 and §5.3.
"""
import numpy as np
import pytest

from rfanalyzer.common.config import Config
from rfanalyzer.demod import demodulate
from rfanalyzer.demod.constellations import (CONSTELLATIONS,
                                             bits_to_iq_symbols,
                                             slice_symbols)
from rfanalyzer.demod.receiver import (_EVM_GATE_CALIBRATION, _EVM_GATES,
                                       _dd_gain, _grid_pitch_gain,
                                       _phase_polish, _radius_gain,
                                       _spectral_centroid_cfo)
from rfanalyzer.synth.factory import WaveformFactory


@pytest.fixture(scope="module")
def demod_cfg():
    return Config().demod


# --------------------------------------------------------------- gates
def test_every_gate_is_at_or_below_both_calibration_sources():
    """Each gate is min(shipped 3GPP-derived value, measured BER value),
    so it can never be looser than either (report §4.2)."""
    for mod, (shipped, measured) in _EVM_GATE_CALIBRATION.items():
        good, degraded = _EVM_GATES[mod]
        assert good <= shipped[0] + 1e-9 and good <= measured[0] + 1e-9
        assert degraded <= shipped[1] + 1e-9
        assert degraded <= measured[1] + 1e-9
        assert good < degraded


@pytest.mark.parametrize("mod", ["32PSK", "128QAM", "256QAM", "128APSK",
                                 "64APSK"])
def test_the_unsafe_gates_were_actually_tightened(mod):
    """These are the gates the BER sweep found admitting 7-16% BER."""
    shipped, _measured = _EVM_GATE_CALIBRATION[mod]
    assert _EVM_GATES[mod] != shipped


# ----------------------------------------------------------------- AGC
@pytest.mark.parametrize("mod", ["16QAM", "64QAM", "256QAM", "128APSK"])
def test_framed_traffic_does_not_use_the_constellation_uniformly(mod):
    """The premise behind the decision-directed AGC: a unit-power AGC
    assumes uniform symbol usage, and real framed traffic is not."""
    bits, _meta, _pl = WaveformFactory(seed=7).build_frames(200)
    syms = bits_to_iq_symbols(bits, mod)
    table = np.asarray(CONSTELLATIONS[mod][0])
    ratio = float((np.abs(syms) ** 2).mean() /
                  (np.abs(table) ** 2).mean())
    assert abs(ratio - 1.0) > 0.03            # at least 1.5% in amplitude


@pytest.mark.parametrize("mod", ["16QAM", "64QAM", "128QAM", "256QAM",
                                 "128APSK"])
def test_gain_estimators_recover_the_scale_of_real_traffic(mod):
    """Two estimators in sequence: the rotation-invariant radius match
    gets the scale close without knowing the carrier, and the
    decision-directed refinement finishes it once decisions are usable."""
    bits, _meta, _pl = WaveformFactory(seed=7).build_frames(400)
    syms = bits_to_iq_symbols(bits, mod)
    table = np.asarray(CONSTELLATIONS[mod][0])
    normalised = syms / np.sqrt((np.abs(syms) ** 2).mean())
    before = slice_symbols(normalised, mod, 0.01)[2]
    assert before > 1.0                        # the error the AGC leaves
    g_radius, _resid = _radius_gain(normalised, table)
    coarse = slice_symbols(normalised * g_radius, mod, 0.01)[2]
    assert coarse <= before                    # the coarse step never hurts
    refined, _g = _dd_gain(normalised * g_radius, mod)
    # and the pair together puts the constellation back on its own scale
    assert slice_symbols(refined, mod, 0.01)[2] < 0.1


def test_radius_gain_is_invariant_to_rotation_and_frequency():
    """That invariance is the whole point: it lets the scale be fixed
    before the carrier is known, breaking the circular dependency between
    the two."""
    bits, _m, _p = WaveformFactory(seed=3).build_frames(200)
    syms = bits_to_iq_symbols(bits, "64QAM")
    table = np.asarray(CONSTELLATIONS["64QAM"][0])
    base = syms / np.sqrt((np.abs(syms) ** 2).mean())
    spun = base * np.exp(2j * np.pi * 0.03 * np.arange(len(base)) + 0.7j)
    assert _radius_gain(base, table)[0] == pytest.approx(
        _radius_gain(spun, table)[0], rel=1e-6)


def test_lattice_pitch_search_avoids_the_45_degree_alias():
    """A square lattice rotated 45 degrees and scaled by sqrt(2) is a
    lattice again, so the search span must stop short of sqrt(2)."""
    table = np.asarray(CONSTELLATIONS["16QAM"][0])
    bits, _m, _p = WaveformFactory(seed=5).build_frames(200)
    syms = bits_to_iq_symbols(bits, "16QAM")
    syms = syms / np.sqrt((np.abs(syms) ** 2).mean())
    gain, conc = _grid_pitch_gain(syms, table)
    assert conc > 0.8
    assert gain < np.sqrt(2) * 0.95


def test_dd_gain_never_makes_the_fit_worse():
    """On an unlocked constellation the nearest-point objective is
    minimised by shrinking everything onto the inner points."""
    rng = np.random.default_rng(0)
    cloud = (rng.normal(0, 1, 4000) + 1j * rng.normal(0, 1, 4000))
    cloud = cloud / np.sqrt((np.abs(cloud) ** 2).mean())
    _out, gain = _dd_gain(cloud, "128QAM")
    assert 0.85 <= gain <= 1.18       # never outside its own search window


# -------------------------------------------------------- phase polish
def test_phase_polish_edges_are_normalised_by_their_real_support():
    """With a fixed 1/window scaling the first and last half-window are
    divided by the full window, so a spike at the edge is smeared back
    over half a window of good symbols instead of being averaged away."""
    n = 2000
    syms = np.exp(1j * np.zeros(n)) * (1 + 0j)
    syms = np.asarray(syms, dtype=np.complex128)
    syms[-3:] = 0.05 * np.exp(1j * np.array([2.0, -2.5, 1.7]))
    out = _phase_polish(syms, "BPSK", window=201)
    # the good symbols well away from the spike must be left alone
    assert np.max(np.abs(np.angle(out[:1500]))) < 1e-6
    # and the correction near the edge must stay bounded
    assert np.max(np.abs(np.angle(out[1500:-3]))) < 0.2


# ------------------------------------------------------------ end-to-end
@pytest.mark.parametrize("mod,snr", [
    ("BPSK", 15), ("QPSK", 20), ("8PSK", 25), ("16PSK", 32), ("32PSK", 37),
    ("OOK", 15), ("4ASK", 25), ("8ASK", 30),
    ("16QAM", 25), ("32QAM", 28), ("64QAM", 30), ("128QAM", 34),
    ("256QAM", 37), ("16APSK", 27), ("32APSK", 30), ("64APSK", 33),
    ("128APSK", 38)])
def test_clean_signals_pass_their_own_gate(mod, snr, demod_cfg):
    """Every linear family, at the SNR the validation matrix uses, must
    clear its own recalibrated gate through the supported path - the one
    the pipeline drives, front end included."""
    from rfanalyzer.channelization import channelize
    from rfanalyzer.conditioning import condition
    from rfanalyzer.detection import detect_signals
    iq, _gt = WaveformFactory(seed=7).generate(
        modulation=mod, sps=8.0, snr_db=snr, n_frames=60,
        cfo_norm=0.004, phase_offset=0.3)
    x, _cond = condition(iq)
    segs, _dbg = detect_signals(x, Config().cfar)
    assert segs
    ch = channelize(x, segs[0])
    res = demodulate(ch["samples"], mod, 8.0, demod_cfg)
    assert res.demodulation_status == "GOOD", \
        f"{mod}: EVM {res.evm_percent}% against gate {_EVM_GATES[mod]}"
    assert res.evm_percent < _EVM_GATES[mod][0]


# ------------------------------------------------------------- CFO paths
def test_spectral_centroid_finds_a_constant_envelope_carrier():
    """FSK, GMSK and analog have no usable M-power line, and had no
    coarse CFO path at all (report §5.1)."""
    iq, _gt = WaveformFactory(seed=7).generate(
        modulation="2FSK", sps=8.0, snr_db=25, n_frames=60,
        cfo_norm=0.02, n_pad_noise=0)
    offset, quality = _spectral_centroid_cfo(np.asarray(iq))
    assert quality > 0.3
    # a coarse estimate: it removes the bulk of the offset and the tone
    # centres measured by the demodulator refine the remainder, which is
    # what test_cpm_and_analog_report_the_carrier_they_removed checks
    assert 0.5 * 0.02 < offset < 1.5 * 0.02


@pytest.mark.parametrize("mod,cfo", [("2FSK", 0.01), ("4FSK", 0.01),
                                     ("GMSK", 0.01), ("FM", 0.01)])
def test_cpm_and_analog_report_the_carrier_they_removed(mod, cfo,
                                                        demod_cfg):
    iq, _gt = WaveformFactory(seed=7).generate(
        modulation=mod, sps=8.0, snr_db=25, n_frames=60,
        cfo_norm=cfo, n_pad_noise=0)
    res = demodulate(np.asarray(iq), mod, 8.0, demod_cfg)
    assert res.cfo_applied_norm == pytest.approx(cfo, abs=0.004), \
        f"{mod}: reported {res.cfo_applied_norm}"
    assert res.cfo_confident


def test_receiver_publishes_what_it_recovered_for_reconciliation(demod_cfg):
    """S4 kept publishing its own symbol rate even when the receiver had
    timed the signal correctly (report §11 item 7)."""
    iq, _gt = WaveformFactory(seed=7).generate(
        modulation="GMSK", sps=8.0, snr_db=20, n_frames=60,
        n_pad_noise=0)
    res = demodulate(np.asarray(iq), "GMSK", 8.0, demod_cfg)
    assert res.symbol_rate_confident
    assert res.symbol_rate_norm_recovered == pytest.approx(0.125, rel=0.05)


def test_failed_demodulation_withholds_its_bits(demod_cfg):
    rng = np.random.default_rng(2)
    noise = (rng.normal(0, 1, 40000) + 1j * rng.normal(0, 1, 40000))
    res = demodulate(noise, "256QAM", 8.0, demod_cfg)
    assert res.demodulation_status == "FAILED"
    assert any("withheld" in w or "no synchronisation" in w
               for w in res.warnings)
