"""S4/S6 reconciliation (report §11 item 7).

WAV-03 reported a 74% symbol-rate error for a GMSK signal whose own
demodulator had timed it correctly, because S4's published figure was
never revisited once S6 measured a better one.
"""
import numpy as np
import pytest

from rfanalyzer.common.models import (Estimate, EstimateState,
                                      DemodulationResult, SignalParameters)
from rfanalyzer.params.estimators import reconcile_with_receiver


def _params(rs=0.05, cfo=-0.02, snr=6.0, snr_state="saturated"):
    p = SignalParameters(sample_rate=48000.0)
    p.symbol_rate_norm = rs
    p.samples_per_symbol = 1.0 / rs
    p.carrier_offset_norm = cfo
    p.snr_db = snr
    p.snr_state = snr_state
    p.symbol_rate_norm_s4 = rs
    p.carrier_offset_norm_s4 = cfo
    p.snr_db_s4 = snr
    return p


def _demod(**kw):
    res = DemodulationResult(modulation="GMSK", samples_per_symbol=8.0)
    res.demodulation_status = "GOOD"
    res.timing_locked = res.carrier_locked = True
    for k, v in kw.items():
        setattr(res, k, v)
    return res


def test_receiver_symbol_rate_replaces_a_wrong_front_end_value():
    p = _params(rs=0.05)
    notes = reconcile_with_receiver(
        p, _demod(symbol_rate_norm_recovered=0.125,
                  symbol_rate_confident=True))
    assert p.symbol_rate_norm == pytest.approx(0.125)
    assert p.samples_per_symbol == pytest.approx(8.0)
    assert p.symbol_rate_hz == pytest.approx(0.125 * 48000.0)
    assert p.symbol_rate_norm_s4 == pytest.approx(0.05)   # kept visible
    assert any(n["quantity"] == "symbol_rate_norm" for n in notes)
    assert p.estimates["symbol_rate"]["verdict"] == "validated"


def test_an_unconfident_receiver_rate_is_not_adopted():
    p = _params(rs=0.05)
    reconcile_with_receiver(
        p, _demod(symbol_rate_norm_recovered=0.125,
                  symbol_rate_confident=False))
    assert p.symbol_rate_norm == pytest.approx(0.05)


def test_agreeing_values_produce_no_reconciliation_note():
    p = _params(rs=0.125, cfo=0.004)
    notes = reconcile_with_receiver(
        p, _demod(symbol_rate_norm_recovered=0.125,
                  symbol_rate_confident=True,
                  cfo_applied_norm=0.004, cfo_confident=True))
    assert not [n for n in notes if n["quantity"] != "snr_db"]


def test_evm_snr_replaces_only_a_non_measurement():
    saturated = _params(snr=6.0, snr_state="saturated")
    reconcile_with_receiver(saturated, _demod(evm_percent=3.0))
    assert saturated.snr_state == "valid"
    assert saturated.snr_db == pytest.approx(30.5, abs=0.2)

    measured = _params(snr=28.0, snr_state="valid")
    reconcile_with_receiver(measured, _demod(evm_percent=3.0))
    assert measured.snr_db == pytest.approx(28.0)   # left alone
    assert "snr_evm" in measured.estimates          # but still recorded


def test_a_failed_demodulation_reconciles_nothing():
    p = _params(rs=0.05)
    res = _demod(symbol_rate_norm_recovered=0.125,
                 symbol_rate_confident=True)
    res.demodulation_status = "FAILED"
    assert reconcile_with_receiver(p, res) == []
    assert p.symbol_rate_norm == pytest.approx(0.05)
