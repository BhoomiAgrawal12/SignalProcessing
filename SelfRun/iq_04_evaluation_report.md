# IQ-04 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `iq_04.iq` |
| Ground truth | `iq_04_truth.json` |
| Analyzer output | `analysis(9).json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | Raw IQ |
| Modulation | 2FSK |
| Sample rate | 900,000 Hz |
| Samples per symbol | 8.0 |
| Symbol rate | 112,500 Bd |
| Normalized symbol rate | 0.125 |
| SNR | 15.0 dB |
| CFO | +0.005 normalized |
| FSK deviation | 0.125 normalized |
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
| Format confidence | — | 0.208 | low | ⚠️ Low confidence |
| Sample rate | 900,000 Hz | 900,000 Hz | 0 | ✅ Correct* |
| Modulation | 2FSK | 2FSK | — | ✅ Correct |
| Modulation confidence | — | 0.904 | — | ✅ Strong |
| FSK tone count | 2 | 2 | 0 | ✅ Exact |
| Samples/symbol | 8.0 | 7.999731 | 0.0034% | ✅ Correct |
| Symbol rate | 112,500 Bd | 112,503.789 Bd | 0.0034% | ✅ Correct |
| SNR | 15.0 dB | 15.0 dB | +0.0 dB | ✅ Exact |
| FSK deviation | 0.125000 | 0.125396 | 0.317% | ✅ Excellent |
| Coarse channel center | +0.005000 | +0.006104 | 22.07% | ❌ Inaccurate |
| Combined CFO | +0.005000 | -0.057499 | 1250.0% | ❌ Wrong |
| Roll-off / excess BW** | 0.35 | 1.00 | 185.7% | ❌ Mismatch |
| FEC | None | none | — | ✅ Correct |
| Interleaver | None | none | — | ✅ Correct |
| Scrambler | None | none | — | ✅ Correct |
| Frame length | 96 bits | 96 bits | 0 | ✅ Correct |
| Sync word | `eb90` | `eb9000` | Extra `00` byte | ⚠️ Partial |
| CRC | CRC-16-CCITT-FALSE | CRC-16-CCITT-FALSE | — | ✅ Correct |

\*Raw IQ sample rate was supplied during analysis.
\**Only directly comparable if `excess_bandwidth` is intended to estimate generator roll-off.

---

## Raw-IQ Ingestion

| Metric | Result | Verdict |
|---|---:|---|
| Input format | raw_iq | ✅ |
| Selected datatype | complex64 little-endian | ✅ Correct |
| Format confidence | 0.208 | ⚠️ Low |
| Sample-rate source | `user` | ✅ Expected |
| Sample rate | 900,000 Hz | ✅ Correct |

---

## Modulation and Demodulation

| Metric | Result | Verdict |
|---|---:|---|
| Modulation prediction | 2FSK | ✅ Correct |
| Confidence | 0.904 | ✅ Strong |
| Detected tone count | 2 | ✅ Exact |
| Demodulation status | GOOD | ✅ |
| Carrier lock | True | ✅ |
| Timing lock | True | ✅ |
| Demodulated symbols | 7,424 | — |
| Demodulated bits | 7,424 | — |
| Tone-fit cost | 0.00748 | ✅ Low |

---

## Parameter Estimation

| Metric | Ground Truth | Analyzer | Error | Verdict |
|---|---:|---:|---:|---|
| SNR | 15.0 dB | 15.0 dB | 0 dB | ✅ Exact |
| Symbol rate | 112,500 Bd | 112,503.789 Bd | 0.0034% | ✅ |
| Samples/symbol | 8.0 | 7.999731 | 0.0034% | ✅ |
| FSK tone count | 2 | 2 | 0 | ✅ |
| FSK deviation | 0.125000 | 0.125396 | 0.317% | ✅ |
| Selected center | +0.005000 | +0.006104 | 22.07% | ❌ |
| Residual CFO | ≈ 0 after channelisation | -0.063603 | large | ❌ |
| Combined CFO | +0.005000 | -0.057499 | 1250.0% | ❌ |

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
| CRC passes | — | 64/64 | ✅ 100% on tested frames |

---

## Payload Recovery

| Metric | Result |
|---|---:|
| Generated payloads | 80 |
| Raw recovered payload records | 76 |
| Correct recovered payloads | 76/76 |
| Raw payload accuracy | **100.0%** |
| Raw payload coverage | **95.0%** |
| Recovered truth indices | 1–76 |
| CRC-valid frames | 64 |
| CRC pass rate | **100.0%** |
| CRC-validated frame coverage | **80.0%** |
| Reported `n_messages` | 76 |
| Actual message-array entries | 64 |
| Correct message payloads | 64/64 |
| Message payload accuracy | **100.0%** |

---

## Internal Reporting Issues

| Issue | Observed Values | Verdict |
|---|---|---|
| CFO reporting | True +0.005000; selected center +0.006104; residual -0.063603 | ❌ Inaccurate |
| Sync word length | Truth `eb90`, analyzer `eb9000` | ⚠️ Overestimated |
| Message count mismatch | `n_messages = 76` but message array contains 64 entries | ❌ Inconsistent |
| Raw-IQ sniff confidence | Correct complex64 selected at 0.208 confidence | ⚠️ Low |

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Modulation accuracy | ✅ Correct |
| Modulation confidence | **0.904** |
| Symbol-rate error | **0.0034%** |
| Samples/symbol error | **0.0034%** |
| SNR error | **0 dB** |
| FSK-deviation error | **0.317%** |
| Combined CFO error | **1250.0%** |
| Raw payload accuracy | **100.0%** |
| Raw payload coverage | **95.0%** |
| CRC-validated coverage | **80.0%** |
| CRC pass rate | **100.0%** |

---

## Final Verdict

| Area | Result |
|---|---|
| Raw-IQ ingestion | ✅ Correct, low sniff confidence |
| 2FSK identification | ✅ Strong |
| Demodulation | ✅ GOOD |
| Symbol-rate estimation | ✅ Excellent |
| SNR estimation | ✅ Exact |
| Tone-count estimation | ✅ Exact |
| FSK-deviation estimation | ✅ Excellent |
| CFO estimation | ❌ Inaccurate |
| FEC / interleaver / scrambler | ✅ Correct |
| Framing | ✅ Mostly correct |
| CRC | ✅ Correct |
| Payload correctness | ✅ 100% |
| Payload coverage | ✅ 95% raw |
| Report consistency | ⚠️ Needs fixes |
| **Overall** | **PASS WITH ISSUES** |

The analyzer performs strongly on this lower-SNR 2FSK raw-IQ case. It correctly identifies 2FSK, recovers the two-tone structure, estimates SNR and symbol rate accurately, demodulates successfully, recovers framing/CRC, and reproduces every recovered application payload exactly. The main weaknesses are CFO estimation, low raw-format sniff confidence, sync-word overextension, and the recurring mismatch between `n_messages` and the actual message array.
