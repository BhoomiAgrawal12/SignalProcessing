# Final Combined WAV Evaluation Report

## Test Set

| File | Modulation / Type | Main Configuration | Final Verdict |
|---|---|---|---|
| WAV-01 | QPSK | 1 MHz, 25 dB, no FEC/interleaver/scrambler | **PASS WITH ISSUES** |
| WAV-02 | BPSK | 2 MHz, 18 dB, convolutional FEC + 8×16 block interleaver + PN9 | **PASS WITH ISSUES** |
| WAV-03 | GMSK | 1.5 MHz, 14 dB, LDPC 128/256 | **PARTIAL / DOWNSTREAM FAILURE** |
| WAV-04 | 4FSK | 800 kHz, 16 dB, no FEC/interleaver/scrambler | **PASS WITH ISSUES** |
| WAV-05 | FM | 480 kHz, 22 dB, analog transmission | **FAIL: MODULATION CLASSIFICATION** |

---

## Per-File Results

| Metric | WAV-01 QPSK | WAV-02 BPSK | WAV-03 GMSK | WAV-04 4FSK | WAV-05 FM |
|---|---:|---:|---:|---:|---:|
| Modulation | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ❌ UNKNOWN |
| Modulation confidence | 0.50 | 0.50 | 1.00 | 0.929 | 0.00 |
| Final SNR error | +7.8 dB | +8.7 dB | 0 dB | 0 dB | +1.2 dB |
| Symbol-rate error | 0% | 0% | ❌ 74.1% at S4* | 0.007% | N/A |
| CFO error | 0.0767% | 0.0105% | 625% | 1250% | 231.09% |
| FEC / bit-layer | ✅ Correct | ✅ Correct | ❌ LDPC not recovered | ✅ Correct | N/A |
| Frame / CRC | ✅ Recovered | ✅ Recovered | ❌ Not recovered | ✅ Recovered | N/A |
| Recovered payload accuracy | 100% | 100% | N/A | 100% | N/A |
| Payload coverage | 82.5% | 90.0% | 0% | 91.25% | N/A |

\*For WAV-03, the later GMSK demodulator internally recovered the correct symbol rate even though the S4 reported value was wrong.

---

## Overall Accuracy / Error Summary

| Metric | Overall Result |
|---|---:|
| WAV format/sample-rate detection | **5/5 = 100%** |
| Modulation classification | **4/5 = 80%** |
| Digital modulation classification only | **4/4 = 100%** |
| Final SNR mean absolute error | **3.54 dB** |
| Final SNR median absolute error | **1.2 dB** |
| Digital S4 symbol rate within 1% | **3/4 = 75%** |
| CFO within 1% of truth | **2/5 = 40%** |
| Digital cases reaching frame/payload recovery | **3/4 = 75%** |
| Correct recovered payload records | **211/211 = 100%** |
| Overall digital payload coverage | **211/320 = 65.9%** |
| CRC-valid frames recovered | **192/320 = 60% coverage** |
| CRC pass rate on tested/recovered frames | **192/192 = 100%** |

---

## Main Findings

| Area | Finding |
|---|---|
| Modulation classification | Strong for QPSK, BPSK, GMSK and 4FSK; **FM failed completely** |
| Demodulation | Strong on all four correctly classified digital modulations |
| Symbol rate | Accurate except GMSK S4 estimation; GMSK demodulator itself recovered the correct rate |
| SNR | Exact on GMSK/4FSK and reasonable on FM, but significantly overestimated on QPSK/BPSK |
| CFO | Excellent on QPSK/BPSK, **poor on GMSK, 4FSK and FM** |
| Bit-layer recovery | Strong on QPSK/BPSK/4FSK; **LDPC chain failed on GMSK** |
| Framing / CRC | Successful for QPSK/BPSK/4FSK; CRC pass rate was 100% on recovered frames |
| Payload correctness | **100% correct wherever payloads were recovered** |
| Payload coverage | Incomplete across digital tests, especially GMSK where recovery was 0% |
| Reporting consistency | Repeated issues with sync-word overextension and some internal count/SNR inconsistencies |

---

## Final Assessment

The analyzer performs well on the tested **digital PSK and 4FSK signals**, with strong modulation recognition, demodulation, framing, CRC validation and exact payload recovery for recovered frames. Its main weaknesses are **CFO estimation outside the PSK cases, SNR consistency, GMSK downstream LDPC recovery, incomplete payload coverage, and analog FM classification**.

### Overall Result: **Mixed but promising**

- **3/5 files:** pass with issues
- **1/5 file:** partial downstream failure
- **1/5 file:** modulation-classification failure
- **Recovered payload accuracy:** 100%
- **Overall modulation accuracy:** 80%
