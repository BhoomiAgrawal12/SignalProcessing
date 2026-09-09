"""S5: receiver-trial arbitration and the ranked trial table (report §8).

The shipped classifier printed only the winner, capped the fused-prior
list at five and trialled only the top three, so a truth outside the top
three was never measured and the analyst could not tell that from a
truth that was measured and rejected.  It also returned early for FSK and
GMSK, before any receiver ran at all (report §5.4).
"""
import numpy as np
import pytest

from rfanalyzer.channelization import channelize
from rfanalyzer.common.config import Config
from rfanalyzer.common.models import TrialRank
from rfanalyzer.conditioning import condition
from rfanalyzer.detection import detect_signals
from rfanalyzer.modulation import classify_modulation
from rfanalyzer.params import estimate_parameters
from rfanalyzer.synth.factory import WaveformFactory

# informationally inseparable pairs, as the validation matrix defines them
EQUIV = {("QPSK", "OQPSK"), ("OQPSK", "QPSK"), ("GMSK", "2FSK"),
         ("2FSK", "GMSK"), ("OOK", "BPSK"), ("BPSK", "OOK")}


@pytest.fixture(scope="module")
def cfg():
    c = Config()
    c.modulation.cvnet_enabled = False      # deterministic and fast
    return c


def _classify(mod, snr, cfg, rank_all=False, max_segments=3, seed=7):
    """Drive S1-S5 the way the pipeline does: down the RANKED segment
    list, not committed to segments[0]."""
    iq, _gt = WaveformFactory(seed=seed).generate(
        modulation=mod, sps=8.0, snr_db=snr, n_frames=60,
        cfo_norm=0.004, phase_offset=0.3)
    x, _c = condition(iq)
    segs, _d = detect_signals(x, cfg.cfar)
    assert segs
    out = None
    for seg in segs[:max_segments]:
        ch = channelize(x, seg)
        p = estimate_parameters(ch["samples"], cfg.params,
                                analysis_band_norm=ch["analysis_band_norm"])
        m = classify_modulation(ch["samples"], p, cfg.modulation,
                                rank_all=rank_all)
        out = (m, p, seg)
        if m.prediction != "UNKNOWN":
            break
    return out


@pytest.mark.parametrize("mod,snr", [
    ("BPSK", 15), ("QPSK", 20), ("OQPSK", 22), ("8PSK", 25),
    ("16PSK", 32), ("32PSK", 37), ("OOK", 15), ("4ASK", 25), ("8ASK", 30),
    ("16QAM", 25), ("32QAM", 28), ("64QAM", 30), ("128QAM", 34),
    ("256QAM", 37), ("16APSK", 27), ("32APSK", 30), ("64APSK", 33),
    ("128APSK", 38), ("2FSK", 15), ("4FSK", 20), ("GMSK", 15)])
def test_blind_top1_classification(mod, snr, cfg):
    m, _p, _seg = _classify(mod, snr, cfg)
    assert m.prediction == mod or (mod, m.prediction) in EQUIV, \
        (f"{mod} classified as {m.prediction}; trial winner "
         f"{m.trial_winner}, prior winner {m.prior_winner}")


@pytest.mark.parametrize("mod,snr", [("2FSK", 10), ("GMSK", 10)])
def test_continuous_phase_signals_reach_a_receiver_trial(mod, snr, cfg):
    """Report §5.4 by name: GMSK at 10 dB used to run as OQPSK at 5.2%
    BER and 2FSK at 10 dB used to stop at UNKNOWN while its own
    demodulator produced a perfect bit stream."""
    m, _p, _seg = _classify(mod, snr, cfg)
    assert m.prediction == mod or (mod, m.prediction) in EQUIV
    cpm = [r for r in m.trial_ranking
           if r.candidate in ("2FSK", "4FSK", "GMSK") and r.measured]
    assert cpm, "no continuous-phase candidate was ever trialled"


def test_trial_ranking_distinguishes_untested_from_rejected(cfg):
    """The distinction the shipped output could not express, and the one
    that made the IQ-02 misclassification invisible (report §8.1)."""
    m, _p, _seg = _classify("16QAM", 25, cfg)
    rows = m.trial_ranking
    assert rows and all(isinstance(r, TrialRank) for r in rows)
    assert {r.candidate for r in rows} == set(cfg.modulation.classes)
    measured = [r for r in rows if r.measured]
    untested = [r for r in rows if not r.measured]
    assert measured and untested
    assert all(r.status == "not measured" and r.score is None
               for r in untested)
    assert all(r.reason for r in untested)
    assert m.trials_run == len(measured)


def test_trial_ranking_is_ordered_by_score_then_prior(cfg):
    m, _p, _seg = _classify("64QAM", 30, cfg)
    scored = [r.score for r in m.trial_ranking if r.score is not None]
    assert scored == sorted(scored, reverse=True)
    # unmeasured rows all sit after the measured ones
    first_unmeasured = next((i for i, r in enumerate(m.trial_ranking)
                             if r.score is None), len(m.trial_ranking))
    assert all(r.score is None for r in m.trial_ranking[first_unmeasured:])


def test_rank_all_measures_every_trialable_candidate(cfg):
    m, _p, _seg = _classify("16QAM", 25, cfg, rank_all=True)
    assert m.trial_mode == "all"
    from rfanalyzer.demod.constellations import MOD_FAMILY
    trialable = {c for c in cfg.modulation.classes
                 if MOD_FAMILY.get(c) is not None}
    measured = {r.candidate for r in m.trial_ranking if r.measured}
    assert trialable <= measured
    truth = next(r for r in m.trial_ranking if r.candidate == "16QAM")
    assert truth.measured and truth.score is not None


def test_prior_and_trial_winners_are_both_reported(cfg):
    m, _p, _seg = _classify("16QAM", 25, cfg)
    assert m.prior_winner and m.trial_winner
    assert m.trial_disagreement == (m.prior_winner != m.trial_winner)
    if m.trial_disagreement:
        assert any("prior" in c for c in m.constraints_applied)


def test_a_sparse_alphabet_is_not_lost_inside_a_denser_one(cfg):
    """16PSK's points ARE sixteen of 32PSK's, so a dense candidate can
    clear the acceptance bar on a sparse signal and stop the search."""
    m, _p, _seg = _classify("16PSK", 32, cfg)
    assert m.prediction == "16PSK"
    rows = {r.candidate: r for r in m.trial_ranking}
    assert rows["16PSK"].measured
    assert rows["16PSK"].occupancy is not None
    if rows["32PSK"].measured and rows["32PSK"].score is not None:
        # the denser table is exercised only half way by a 16PSK signal
        assert rows["32PSK"].occupancy < rows["16PSK"].occupancy


def test_noise_only_input_is_reported_as_unknown(cfg):
    rng = np.random.default_rng(11)
    x = (rng.normal(0, 1, 60000) + 1j * rng.normal(0, 1, 60000))
    x = np.asarray(x, dtype=np.complex64)
    segs, _d = detect_signals(x, cfg.cfar)
    if not segs:
        return                       # nothing detected is also correct
    ch = channelize(x, segs[0])
    p = estimate_parameters(ch["samples"], cfg.params,
                            analysis_band_norm=ch["analysis_band_norm"])
    m = classify_modulation(ch["samples"], p, cfg.modulation)
    assert m.prediction == "UNKNOWN"
    assert any("withheld" in c or "no candidate locks" in c
               for c in m.constraints_applied)


def test_saturated_snr_does_not_force_the_noisy_classifier_mode(cfg):
    """cumulants.py widens its posterior below 15 dB and the shipped S4
    estimate never reached 15 dB for QAM, so the sharp mode was
    unreachable for exactly the constellations that need it
    (report §3.3)."""
    from rfanalyzer.modulation.cumulants import classify_cumulants
    from rfanalyzer.demod.constellations import CONSTELLATIONS
    rng = np.random.default_rng(5)
    table = np.asarray(CONSTELLATIONS["16QAM"][0])
    table = table / np.sqrt((np.abs(table) ** 2).mean())
    syms = table[rng.integers(0, len(table), 40000)]
    noise = rng.normal(0, np.sqrt(10 ** (-28 / 10) / 2), (len(syms), 2))
    syms = syms + noise[:, 0] + 1j * noise[:, 1]
    sharp = classify_cumulants(syms, None)          # state not trusted
    blurred = classify_cumulants(syms, 6.7)         # the saturated value
    assert max(sharp["probabilities"].values()) > \
        max(blurred["probabilities"].values())
