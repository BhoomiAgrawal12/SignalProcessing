"""S4 measurement layer: value + confidence + method + validity STATE.

These cover report §3 (the SNR estimator that saturated silently), §5
defects D3 and D4 (excess bandwidth and a missing OBW) and the §9 rule
that a measurement must be able to say it did not measure.
"""
import numpy as np
import pytest
from scipy import signal as sg

from rfanalyzer.common.models import EstimateState
from rfanalyzer.demod.constellations import CONSTELLATIONS
from rfanalyzer.demod.filters import rrc_taps
from rfanalyzer.params.estimators import (excess_bandwidth, m2m4_ceiling_db,
                                          m2m4_snr, occupied_bandwidth,
                                          signal_kurtosis, snr_from_evm,
                                          spectral_snr, symbol_rate)


def _symbols(mod, n, rng):
    table = np.asarray(CONSTELLATIONS[mod][0], dtype=np.complex128)
    table = table / np.sqrt((np.abs(table) ** 2).mean())
    return table[rng.integers(0, len(table), n)]


def _shaped(mod, n_sym, sps, snr_db, rng, beta=0.35):
    syms = _symbols(mod, n_sym, rng)
    up = np.zeros(len(syms) * int(sps), dtype=np.complex128)
    up[::int(sps)] = syms
    x = sg.fftconvolve(up, rrc_taps(int(sps), 10, beta), mode="same")
    x = x / np.sqrt((np.abs(x) ** 2).mean())
    noise = rng.normal(0, np.sqrt(10 ** (-snr_db / 10) / 2), (len(x), 2))
    return x + noise[:, 0] + 1j * noise[:, 1]


# ---------------------------------------------------------------- kurtosis
def test_constant_modulus_alphabets_have_unit_kurtosis():
    for mod in ("BPSK", "QPSK", "8PSK", "16PSK", "32PSK"):
        assert signal_kurtosis(mod) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("mod,expected_ceiling", [
    ("16QAM", 6.72), ("64QAM", 5.67), ("16APSK", 8.84)])
def test_m2m4_ceiling_matches_the_measured_saturation(mod, expected_ceiling):
    """The analytic ceiling of a ka=1 estimator reproduces the ceilings
    the review measured empirically (report §3.2), to within 0.1 dB."""
    got = m2m4_ceiling_db(signal_kurtosis(mod))
    assert got == pytest.approx(expected_ceiling, abs=0.05)


def test_constant_modulus_signal_has_no_ceiling():
    assert m2m4_ceiling_db(1.0) == float("inf")


# --------------------------------------------------------------------- SNR
def test_m2m4_uses_the_kurtosis_it_is_given(rng):
    """The shipped estimator accepted kurtosis_signal and ignored it, so
    16QAM read ~6.7 dB at every SNR above 15 (report §3.1)."""
    syms = _symbols("16QAM", 200000, rng)
    noise = rng.normal(0, np.sqrt(10 ** (-28 / 10) / 2), (len(syms), 2))
    x = syms + noise[:, 0] + 1j * noise[:, 1]

    blind = m2m4_snr(x)                       # constant-modulus assumption
    informed = m2m4_snr(x, signal_kurtosis("16QAM"))
    assert blind.value < 8.0                  # saturated near the 6.7 ceiling
    assert informed.value > 22.0              # and released once ka is right


def test_m2m4_reports_saturation_against_an_independent_reference(rng):
    x = _shaped("16QAM", 40000, 8, 28, rng)
    ref = spectral_snr(x)
    assert ref.usable
    est = m2m4_snr(x, reference_snr_db=ref.detail["full_band_snr_db"])
    assert est.state is EstimateState.SATURATED
    assert not est.usable
    assert est.lower_bound is not None        # still a usable bound
    assert "floor" in est.detail["reason"]


@pytest.mark.parametrize("mod", ["QPSK", "16QAM", "64QAM", "256QAM"])
@pytest.mark.parametrize("snr_db", [5, 15, 25])
def test_spectral_snr_is_accurate_and_kurtosis_free(mod, snr_db, rng):
    """Es/N0 = full-band SNR + 10log10(sps) for a Nyquist system."""
    x = _shaped(mod, 30000, 8, snr_db, rng)
    est = spectral_snr(x, reference_bandwidth=0.125)
    assert est.usable
    assert est.value == pytest.approx(snr_db + 10 * np.log10(8), abs=2.5)


def test_spectral_snr_is_unresolvable_on_noise_only_data(rng):
    """There is no signal to measure an SNR of, and saying "-14 dB" would
    dress that up as a measurement."""
    x = (rng.normal(0, 1, 200000) + 1j * rng.normal(0, 1, 200000))
    est = spectral_snr(x)
    assert est.state is EstimateState.UNRESOLVABLE
    assert est.value is None


def test_spectral_snr_is_unresolvable_when_the_signal_fills_the_band(rng):
    """A Nyquist-rate signal leaves no noise-only region, so there is no
    floor to read - and the estimator must say so rather than guess."""
    syms = _symbols("QPSK", 200000, rng)          # one sample per symbol
    noise = rng.normal(0, np.sqrt(10 ** (-30 / 10) / 2), (len(syms), 2))
    x = syms + noise[:, 0] + 1j * noise[:, 1]
    est = spectral_snr(x)
    assert est.state is EstimateState.UNRESOLVABLE
    assert est.value is None
    assert "reason" in est.detail


def test_snr_from_evm_is_es_n0():
    assert snr_from_evm(10.0).value == pytest.approx(20.0, abs=0.05)
    assert snr_from_evm(0).state is EstimateState.UNRESOLVABLE


# ------------------------------------------------- bandwidth and roll-off
@pytest.mark.parametrize("beta", [0.2, 0.35, 0.5])
def test_excess_bandwidth_recovers_the_true_rolloff(beta, rng):
    """Defect D3: the shipped estimate was obw99/Rs - 1, which read 0.14
    for a true 0.35 because it measures where the 99% integral stops, not
    where the filter rolls off."""
    x = _shaped("QPSK", 40000, 8, 30, rng, beta=beta)
    bw = occupied_bandwidth(x)
    got = excess_bandwidth(bw["psd_freq"], bw["psd_lin"], 0.125,
                           bw["f_center"])
    assert got["rolloff"] == pytest.approx(beta, abs=0.05)


def test_occupied_bandwidth_never_returns_none_for_a_real_signal(rng):
    """Defect D4: OBW came back None whenever the noise-floor subtraction
    emptied the spectrum."""
    for snr in (0, 15, 40):
        x = _shaped("QPSK", 20000, 8, snr, rng)
        bw = occupied_bandwidth(x)
        assert bw["obw99"] is not None
        assert bw["obw99"] > 0
        assert "method" in bw


def test_occupied_bandwidth_reports_a_method_even_when_it_retreats(rng):
    x = np.full(20000, 1.0 + 0j)          # a bare carrier: no noise floor
    bw = occupied_bandwidth(x)
    assert bw["method"]


# ------------------------------------------------------------ symbol rate
@pytest.mark.parametrize("sps", [4, 8, 20, 40])
def test_symbol_rate_across_samples_per_symbol(sps, rng):
    """Audio-rate telemetry runs at 40 sps and higher; the shipped search
    floor of 0.25*OBW99 sat uncomfortably close to those rates."""
    x = _shaped("QPSK", 20000, sps, 25, rng)
    bw = occupied_bandwidth(x)
    got = symbol_rate(x, 1e-4, obw99=bw["obw99"])
    assert got["candidates"]
    assert got["candidates"][0]["rate_norm"] == pytest.approx(
        1.0 / sps, rel=0.02)


def test_symbol_rate_reports_its_search_floor(rng):
    x = _shaped("QPSK", 20000, 8, 25, rng)
    got = symbol_rate(x, 1e-4, obw99=0.16)
    assert "min_rate" in got
    assert got["min_rate"] == pytest.approx(0.2 * 0.16)
