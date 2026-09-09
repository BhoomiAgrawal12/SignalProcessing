# IQ-03 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `iq_03.iq` |
| Ground truth | `iq_03_truth.json` |
| Analyzer output | `analysis(8).json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | Raw IQ |
| Modulation | 8ASK |
| Sample rate | 1,200,000 Hz |
| Samples per symbol | 8.0 |
| Symbol rate | 150,000 Bd |
| Normalized symbol rate | 0.125 |
| SNR | 20.0 dB |
| CFO | +0.005 normalized |
| Roll-off | 0.35 |
| FEC | None |
| Interleaver | None |
| Scrambler | None |
| Frame length | 96 bits |
| Sync word | `eb90` |
| CRC | CRC-16-CCITT-FALSE |
| Payload length | 6 bytes/frame |
| Generated payloads | 80 |

---

## Ground Truth vs Analyzer

| Parameter | Ground Truth | Analyzer Result | Error / Difference | Verdict |
|---|---:|---:|---:|---|
| Format | Raw IQ | raw_iq | — | ✅ Correct |
| Datatype | complex64 | complex64 | — | ✅ Correct |
| Format confidence | — | 0.223 | Low | ⚠️ Low confidence |
| Sample rate | 1,200,000 Hz | 1,200,000 Hz | 0 | ✅ Correct* |
| Modulation | 8ASK | 8ASK | — | ✅ Correct |
| Modulation confidence | — | 0.668 | 4ASK alternative 0.332 | ✅ Acceptable |
| Samples/symbol | 8.0 | 8.000779 | 0.0097% | ✅ Correct |
| Symbol rate | 150,000 Bd | 149,985.404 Bd | 0.0097% | ✅ Correct |
| SNR | 20.0 dB | 24.7 dB | +4.7 dB | ❌ Overestimated |
| Reconstructed CFO | +0.005000 | +0.005008 | 0.1592% | ✅ Correct |
| Roll-off / excess BW** | 0.35 | 0.15 | 57.1% | ❌ Mismatch |
| OFDM | False | Detected, FFT 256 / CP 32 | False positive | ❌ Incorrect |
| FEC | None | None | — | ✅ Correct |
| Interleaver | None | None | — | ✅ Correct |
| Scrambler | None | None | — | ✅ Correct |
| Frame length | 96 bits | 96 bits | 0 | ✅ Correct |
| Sync word | `eb90` | `eb9000` | Extra `00` byte | ⚠️ Partial |
| CRC | CRC-16-CCITT-FALSE | CRC-16-CCITT-FALSE | — | ✅ Correct |

\*Raw IQ sample rate was supplied during analysis.
\**Only directly comparable if `excess_bandwidth` is intended to estimate generator roll-off.

---

## Raw-IQ Ingestion

| Metric | Result | Verdict |
|---|---:|---|
| Format | raw_iq | ✅ |
| Selected datatype | complex64 little-endian | ✅ |
| Format confidence | 0.223 | ⚠️ Low |
| Sample-rate source | `user` | ✅ Expected |
| Sample rate | 1,200,000 Hz | ✅ Correct |

---

## Modulation and Demodulation

| Metric | Result | Verdict |
|---|---:|---|
| Prediction | 8ASK | ✅ Correct |
| Confidence | 0.668 | ✅ |
| Alternative | 4ASK at 0.332 | ⚠️ Some ambiguity |
| Demodulation status | GOOD | ✅ |
| EVM | 5.8% | ✅ |
| Carrier lock | True | ✅ |
| Timing lock | True | ✅ |
| Demodulated symbols | 2,311 | — |
| Demodulated bits | 6,933 | — |

---

## Parameter Estimation

| Metric | Ground Truth | Analyzer | Error | Verdict |
|---|---:|---:|---:|---|
| Symbol rate | 150,000 Bd | 149,985.404 Bd | 0.0097% | ✅ |
| Samples/symbol | 8.0 | 8.000779 | 0.0097% | ✅ |
| SNR | 20.0 dB | 24.7 dB | +4.7 dB | ❌ |
| Selected-segment SNR | 20.0 dB | 28.14 dB | +8.14 dB | ❌ |
| CFO | +0.005000 | +0.005008 reconstructed | 0.1592% | ✅ |
| Roll-off / excess BW | 0.35 | 0.15 | 57.1% | ❌ |

---

## OFDM False Positive

| Property | Ground Truth | Analyzer | Verdict |
|---|---|---|---|
| OFDM present | No | Yes | ❌ False positive |
| FFT size | N/A | 256 | ❌ Spurious |
| CP length | N/A | 32 | ❌ Spurious |
| OFDM confidence | N/A | 1.0 | ❌ Overconfident false detection |

The same report correctly classifies the waveform as 8ASK while independently claiming OFDM with confidence 1.0. This is an internal semantic inconsistency.

---

## Bit-Layer Evaluation

| Layer | Ground Truth | Analyzer | Verdict |
|---|---|---|---|
| Scrambler | None | None | ✅ Correct |
| Interleaver | None | None | ✅ Correct |
| FEC | None | None | ✅ Correct |

---

## Framing and CRC

| Parameter | Ground Truth | Analyzer | Verdict |
|---|---|---|---|
| Frame length | 96 bits | 96 bits | ✅ Correct |
| Sync word | `eb90` | `eb9000` | ⚠️ Extra byte |
| CRC | CRC-16-CCITT-FALSE | CRC-16-CCITT-FALSE | ✅ Correct |
| CRC passes | — | 64/64 | ✅ 100% |

---

## Payload Recovery

| Metric | Result |
|---|---:|
| Generated payloads | 80 |
| Raw recovered records | 72 |
| Correct recovered payloads | 72/72 |
| Raw payload accuracy | **100%** |
| Raw payload coverage | **90%** |
| Recovered truth indices | 8–79 |
| CRC-valid frames | 64 |
| CRC pass rate | **100%** |
| CRC-validated coverage | **80%** |
| Reported `n_messages` | 72 |
| Actual message-array entries | 64 |
| Correct message payloads | 64/64 |
| Message payload accuracy | **100%** |

---

## Internal Reporting Issues

| Issue | Observed Values | Verdict |
|---|---|---|
| OFDM contradiction | 8ASK correctly classified, but OFDM also flagged with confidence 1.0 | ❌ Inconsistent |
| Sync word length | Truth `eb90`, analyzer `eb9000` | ⚠️ Overestimated |
| Message count mismatch | `n_messages = 72`, actual message array = 64 | ❌ Inconsistent |
| Raw-IQ sniff confidence | Correct complex64 selected at 0.223 confidence | ⚠️ Low |
| SNR estimates | Truth 20.0 dB, selected 28.14 dB, final 24.7 dB | ❌ Inconsistent / high |

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Modulation accuracy | ✅ Correct |
| Modulation confidence | **0.668** |
| Symbol-rate error | **0.0097%** |
| Samples/symbol error | **0.0097%** |
| CFO error | **0.1592%** |
| SNR error | **+4.7 dB** |
| OFDM false positive | **Yes** |
| Raw payload accuracy | **100%** |
| Raw payload coverage | **90%** |
| CRC-validated coverage | **80%** |
| CRC pass rate | **100%** |

---

## Final Verdict

| Area | Result |
|---|---|
| Raw-IQ ingestion | ✅ Correct, low sniff confidence |
| 8ASK identification | ✅ Correct |
| Demodulation | ✅ GOOD |
| Symbol-rate estimation | ✅ Excellent |
| CFO estimation | ✅ Very good |
| SNR estimation | ❌ Overestimated |
| OFDM detection | ❌ False positive |
| FEC / interleaver / scrambler | ✅ Correct |
| Framing | ✅ Mostly correct |
| CRC | ✅ Correct |
| Payload correctness | ✅ 100% |
| Payload coverage | ✅ 90% raw |
| Report consistency | ⚠️ Needs fixes |
| **Overall** | **PASS WITH ISSUES** |

The analyzer performs well on this 8ASK raw-IQ case: classification, demodulation, symbol timing, CFO, framing, CRC and recovered payload bytes are all strong. The main problems are SNR overestimation, a high-confidence false OFDM detection, low raw-format sniff confidence, sync-word overextension, and message-count inconsistency.
