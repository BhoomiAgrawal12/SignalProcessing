# IQ-01 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `iq_01.iq` |
| Ground truth | `iq_01_truth.json` |
| Analyzer output | `analysis(6).json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | Raw IQ |
| Modulation | 8PSK |
| Sample rate | 2,400,000 Hz |
| Samples per symbol | 8.0 |
| Symbol rate | 300,000 Bd |
| Normalized symbol rate | 0.125 |
| SNR | 24.0 dB |
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
| Format confidence | — | 0.225 | low / ambiguous sniffing | ⚠️ Low confidence |
| Sample rate | 2,400,000 Hz | 2,400,000 Hz | 0 | ✅ Correct* |
| Modulation | 8PSK | 8PSK | — | ✅ Correct |
| Modulation confidence | — | 1.00 | — | ✅ Strong |
| Samples/symbol | 8.0 | 8.001167 | 0.0146% | ✅ Correct |
| Symbol rate | 300,000 Bd | 299,956.249 Bd | 0.0146% | ✅ Correct |
| SNR | 24.0 dB | 25.5 dB | +1.5 dB | ⚠️ Moderate error |
| S4 combined CFO | +0.005000 | -0.008676 | 273.5% | ❌ Wrong |
| Effective S6 CFO | +0.005000 | +0.005003 | 0.0517% | ✅ Excellent |
| Roll-off / excess BW** | 0.35 | 0.15 | 57.1% | ❌ Mismatch |
| FEC | None | none | — | ✅ Correct |
| Interleaver | None | none | — | ✅ Correct |
| Scrambler | None | none | — | ✅ Correct |
| Frame length | 96 bits | 96 bits | 0 | ✅ Correct |
| Sync word | `eb90` | `eb9000` | Extra `00` byte | ⚠️ Partial |
| CRC | CRC-16-CCITT-FALSE | CRC-16-CCITT-FALSE | — | ✅ Correct |

\*Raw IQ sample rate was supplied by the user during analysis; it was not inferred from the file.
\**Only score roll-off against `excess_bandwidth` if those fields are intended to represent the same quantity.

---

## Raw-IQ Ingestion

| Metric | Result | Verdict |
|---|---:|---|
| Input format | raw_iq | ✅ |
| Datatype selected | complex64 little-endian | ✅ Correct |
| Format confidence | 0.225 | ⚠️ Low |
| Sample-rate source | `user` | ✅ Expected for raw IQ |
| Sample rate | 2,400,000 Hz | ✅ Correct |

---

## Modulation and Demodulation

| Metric | Result | Verdict |
|---|---:|---|
| Modulation prediction | 8PSK | ✅ Correct |
| Confidence | 1.00 | ✅ Strong |
| Demodulation status | GOOD | ✅ |
| EVM | 5.3% | ✅ |
| Carrier lock | True | ✅ |
| Timing lock | True | ✅ |
| Demodulated symbols | 2,313 | — |
| Demodulated bits | 6,939 | — |

---

## Parameter Estimation

| Metric | Ground Truth | Analyzer | Error | Verdict |
|---|---:|---:|---:|---|
| SNR | 24.0 dB | 25.5 dB | +1.5 dB | ⚠️ |
| Symbol rate | 300,000 Bd | 299,956.249 Bd | 0.0146% | ✅ |
| Samples/symbol | 8.0 | 8.001167 | 0.0146% | ✅ |
| Selected channel center | +0.005000 | +0.005859 | — | ⚠️ Coarse |
| S4 residual CFO | ≈ residual after channelization | -0.014535 | large | ❌ |
| Effective S6 CFO reconstruction | +0.005000 | +0.005003 | 0.0517% | ✅ Excellent |

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
| Raw recovered payload records | 72 |
| Correct recovered payloads | 72/72 |
| Raw payload accuracy | **100.0%** |
| Raw payload coverage | **90.0%** |
| Recovered truth indices | 8–79 |
| CRC-valid frames | 64 |
| CRC pass rate | **100.0%** |
| CRC-validated frame coverage | **80.0%** |
| Reported `n_messages` | 72 |
| Actual message-array entries | 64 |
| Correct message payloads | 64/64 |
| Message payload accuracy | **100.0%** |

---

## Internal Reporting Issues

| Issue | Observed Values | Verdict |
|---|---|---|
| CFO inconsistency | S4 combined CFO -0.008676, but S6 effective CFO +0.005003 | ❌ S4 wrong / S6 correct |
| Sync word length | Truth `eb90`, analyzer `eb9000` | ⚠️ Overestimated |
| Message count mismatch | `n_messages = 72` but message array contains 64 | ❌ Inconsistent |
| Raw-IQ sniff confidence | Correct complex64 selected at only 0.225 confidence | ⚠️ Ambiguous |

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Modulation accuracy | ✅ Correct |
| Modulation confidence | **1.00** |
| Symbol-rate error | **0.0146%** |
| Samples/symbol error | **0.0146%** |
| SNR error | **+1.5 dB** |
| S4 CFO error | **273.5%** |
| Effective S6 CFO error | **0.0517%** |
| Raw payload accuracy | **100.0%** |
| Raw payload coverage | **90.0%** |
| CRC-validated frame coverage | **80.0%** |
| CRC pass rate | **100.0%** |

---

## Final Verdict

| Area | Result |
|---|---|
| Raw-IQ ingestion | ✅ Correct, low sniff confidence |
| 8PSK identification | ✅ Excellent |
| Demodulation | ✅ GOOD |
| Symbol-rate estimation | ✅ Excellent |
| Effective CFO correction | ✅ Excellent at S6 |
| S4 CFO reporting | ❌ Incorrect |
| SNR estimation | ⚠️ +1.5 dB error |
| FEC / interleaver / scrambler | ✅ Correct |
| Framing | ✅ Mostly correct |
| CRC | ✅ Correct |
| Payload correctness | ✅ 100% |
| Payload coverage | ✅ 90% raw |
| Report consistency | ⚠️ Needs fixes |
| **Overall** | **PASS WITH ISSUES** |

The analyzer performs strongly on this raw-IQ 8PSK case. It correctly selects the raw complex64 format, classifies 8PSK with full confidence, demodulates successfully, estimates the symbol rate accurately, recovers framing/CRC, and reproduces every recovered application payload exactly. The main issues are low raw-format sniff confidence, an incorrect S4 CFO field despite excellent effective S6 correction, sync-word overextension, moderate SNR error, and message-count inconsistency.
