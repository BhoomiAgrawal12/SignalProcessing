# Final Combined IQ Evaluation Report

## Test Set

| File | Modulation / Type | Main Configuration | Final Verdict |
|---|---|---|---|
| IQ-01 | 8PSK | 2.4 MHz, 24 dB, no FEC/interleaver/scrambler | **PASS WITH ISSUES** |
| IQ-02 | 16QAM | 1.8 MHz, 28 dB, convolutional FEC + 8×16 block interleaver + PN9 | **FAIL: MODULATION MISCLASSIFICATION / DOWNSTREAM FAILURE** |
| IQ-03 | 8ASK | 1.2 MHz, 20 dB, no FEC/interleaver/scrambler | **PASS WITH ISSUES** |
| IQ-04 | 2FSK | 900 kHz, 15 dB, no FEC/interleaver/scrambler | **PASS WITH ISSUES** |
| IQ-05 | 16APSK | 2.0 MHz, 26 dB, LDPC 128/256 | **FAIL: MODULATION MISCLASSIFICATION / DEMODULATION FAILURE** |

---

## Per-File Results

| Metric | IQ-01 8PSK | IQ-02 16QAM | IQ-03 8ASK | IQ-04 2FSK | IQ-05 16APSK |
|---|---:|---:|---:|---:|---:|
| Modulation | ✅ Correct | ❌ 64QAM | ✅ Correct | ✅ Correct | ❌ 128QAM |
| Modulation confidence | 1.000 | 0.400 wrong class | 0.668 | 0.904 | 0.362 wrong class |
| Symbol-rate error | 0.0146% | 0% | 0.0097% | 0.0034% | 0% |
| Final SNR error | +1.5 dB | -9.5 dB | +4.7 dB | 0 dB | -21.4 dB |
| CFO error | 0.0517%* | 0.0977% | 0.1592% | ~1250% | 514.3% |
| FEC / bit-layer | ✅ Correct | ❌ Missed | ✅ Correct | ✅ Correct | ❌ Not reached |
| Frame / CRC | ✅ Recovered | ❌ Not recovered | ✅ Recovered | ✅ Recovered | ❌ Not reached |
| Raw payload accuracy | 100% | N/A | 100% | 100% | N/A |
| Raw payload coverage | 90% | 0% | 90% | 95% | 0% |

*IQ-01 CFO uses the effective S6 correction because the S4 CFO field was internally inconsistent.

---

## Overall Accuracy / Error Summary

| Metric | Overall Result |
|---|---:|
| Raw-IQ format/sample-rate handling | **5/5 = 100%** |
| Modulation classification | **3/5 = 60%** |
| Symbol-rate estimate within 1% | **5/5 = 100%** |
| Final SNR mean absolute error | **7.42 dB** |
| Final SNR median absolute error | **4.7 dB** |
| CFO within 1% of truth | **3/5 = 60%** |
| Cases reaching frame/payload recovery | **3/5 = 60%** |
| Correct recovered payload records | **220/220 = 100%** |
| Overall raw payload coverage | **220/400 = 55%** |
| CRC-valid frame coverage | **192/400 = 48%** |
| CRC pass rate on recovered/tested frames | **192/192 = 100%** |

---

## Main Findings

| Area | Finding |
|---|---|
| Raw IQ ingestion | Correct complex64 input selected in all 5 tests, but sniff confidence remained low (~0.21–0.23) |
| Modulation classification | Strong for 8PSK, 8ASK and 2FSK; failed on **16QAM** and **16APSK** |
| Symbol rate | Very strong across all five IQ files |
| SNR | Mixed; exact/close on some cases but severe errors on 16QAM and especially 16APSK |
| CFO | Strong on 8PSK, 16QAM and 8ASK; poor on 2FSK and 16APSK |
| Bit-layer recovery | Successful when modulation classification was correct |
| Framing / CRC | Successful on IQ-01, IQ-03 and IQ-04 with **100% CRC pass rate** on recovered frames |
| Payload correctness | **100% correct wherever payload was recovered** |
| Payload coverage | Incomplete due to trimming/recovery limits and complete failure on IQ-02/IQ-05 |
| False positives | IQ-03 incorrectly reported OFDM despite correctly identifying 8ASK |
| Quality gating | IQ-05 correctly withheld unreliable bits after demodulation lost carrier lock |
| Reporting consistency | Repeated sync-word overextension (`eb90` → `eb9000`), low raw-format confidence, and some count/CFO inconsistencies |

---

## Final Assessment

The analyzer performs strongly on **8PSK, 8ASK and 2FSK raw-IQ signals**, with excellent symbol-rate estimation, successful demodulation, framing, CRC validation and exact recovered payload bytes.

The main weakness is **higher-order constellation classification**. The 16QAM test was misclassified as 64QAM, while 16APSK was misclassified as 128QAM, causing complete downstream failure in both cases. SNR and CFO estimation also become unreliable in several modulation families.

### Overall Result: **Mixed**

- **3/5 IQ files:** Pass with issues
- **2/5 IQ files:** Fail due to modulation misclassification
- **Modulation classification accuracy:** **60%**
- **Symbol-rate accuracy within 1%:** **100%**
- **Recovered payload accuracy:** **100%**
- **Overall raw payload coverage:** **55%**
- **CRC pass rate on recovered frames:** **100%**
