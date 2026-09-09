# WAV-04 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `wav_04.wav` |
| Ground truth | `wav_04_truth.json` |
| Analyzer output | `analysis(4).json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | WAV |
| Modulation | 4FSK |
| Sample rate | 800,000 Hz |
| Samples per symbol | 8.0 |
| Symbol rate | 100,000 Bd |
| Normalized symbol rate | 0.125 |
| SNR | 16.0 dB |
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
| Format | WAV | WAV | — | ✅ Correct |
| Sample rate | 800,000 Hz | 800,000 Hz | 0 | ✅ Correct |
| Modulation | 4FSK | 4FSK | — | ✅ Correct |
| Modulation confidence | — | 0.929 | — | ✅ Strong |
| FSK tone count | 4 | 4 | 0 | ✅ Correct |
| Samples/symbol | 8.0 | 7.999442 | 0.0070% | ✅ Correct |
| Symbol rate | 100,000 Bd | 100,006.976 Bd | 0.0070% | ✅ Correct |
| SNR | 16.0 dB | 16.0 dB | +0.0 dB | ✅ Exact |
| FSK deviation | 0.125000 | 0.125585 | 0.468% | ✅ Correct |
| Coarse channel center | +0.005000 | +0.007812 | 56.25% | ❌ Inaccurate |
| Combined CFO | +0.005000 | +0.067501 | 1250.0% | ❌ Wrong |
| FEC | None | none | — | ✅ Correct |
| Interleaver | None | none | — | ✅ Correct |
| Scrambler | None | none | — | ✅ Correct |
| Frame length | 96 bits | 96 bits | 0 | ✅ Correct |
| Sync word | `eb90` | `eb9000` | Extra `00` byte | ⚠️ Partial |
| CRC | CRC-16-CCITT-FALSE | CRC-16-CCITT-FALSE | — | ✅ Correct |
| Initial phase offset | 0.4 rad | No direct equivalent | — | ⚪ Not verifiable |

---

## Modulation and Demodulation

| Metric | Result | Verdict |
|---|---:|---|
| Modulation prediction | 4FSK | ✅ Correct |
| Confidence | 0.929 | ✅ Strong |
| Detected tone count | 4 | ✅ Exact |
| Demodulation status | GOOD | ✅ |
| Carrier lock | True | ✅ |
| Timing lock | True | ✅ |
| Demodulated symbols | 3,584 | — |
| Demodulated bits | 7,168 | — |
| Tone-fit cost | 0.00665 | ✅ Low |

---

## Parameter Estimation

| Metric | Ground Truth | Analyzer | Error | Verdict |
|---|---:|---:|---:|---|
| SNR | 16.0 dB | 16.0 dB | 0 dB | ✅ Exact |
| Symbol rate | 100,000 Bd | 100,006.976 Bd | 0.0070% | ✅ |
| Samples/symbol | 8.0 | 7.999442 | 0.0070% | ✅ |
| FSK tone count | 4 | 4 | 0 | ✅ |
| FSK deviation | 0.125000 | 0.125585 | 0.468% | ✅ |
| Selected center | +0.005000 | +0.007812 | 56.25% | ❌ |
| Residual CFO | ≈ 0 after channelization | +0.059688 | large | ❌ |
| Combined CFO | +0.005000 | +0.067501 | 1250.0% | ❌ |

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
| Raw recovered payload records | 73 |
| Correct recovered payloads | 73/73 |
| Raw payload accuracy | **100.0%** |
| Raw payload coverage | **91.25%** |
| Recovered truth indices | 1–73 |
| CRC-valid frames | 64 |
| CRC pass rate on tested frames | **100.0%** |
| CRC-validated frame coverage | **80.0%** |
| Message-array entries | 64 |
| Correct message payloads | 64/64 |
| Message payload accuracy | **100.0%** |

---

## Internal Reporting Issues

| Issue | Observed Values | Verdict |
|---|---|---|
| Message count mismatch | `n_messages = 73` but message array contains 64 entries | ❌ Inconsistent |
| Sync word length | Truth `eb90`, analyzer `eb9000` | ⚠️ Overestimated |
| CFO reporting | True +0.005; selected center +0.007812; residual +0.059688 | ❌ Inconsistent / inaccurate |

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Modulation accuracy | ✅ Correct |
| Modulation confidence | **0.929** |
| Symbol-rate error | **0.0070%** |
| Samples/symbol error | **0.0070%** |
| SNR error | **0 dB** |
| FSK-deviation error | **0.468%** |
| Coarse center error | **56.25%** |
| Combined CFO error | **1250.0%** |
| Raw payload accuracy | **100.0%** |
| Raw payload coverage | **91.25%** |
| CRC-validated frame coverage | **80.0%** |
| CRC pass rate | **100.0%** |

---

## Final Verdict

| Area | Result |
|---|---|
| Format detection | ✅ Correct |
| 4FSK identification | ✅ Correct |
| SNR estimation | ✅ Exact |
| Symbol-rate estimation | ✅ Excellent |
| Tone-count estimation | ✅ Exact |
| FSK-deviation estimation | ✅ Very accurate |
| Demodulation | ✅ GOOD |
| FEC / interleaver / scrambler | ✅ Correct |
| Framing | ✅ Mostly correct |
| CRC | ✅ Correct |
| Payload correctness | ✅ 100% on recovered records |
| Payload coverage | ✅ 91.25% raw |
| CFO estimation | ❌ Inaccurate |
| Report consistency | ⚠️ Needs fixes |
| **Overall** | **PASS WITH ISSUES** |

The analyzer performs strongly on this 4FSK case. It correctly identifies the modulation, tone count, SNR, symbol rate, FSK deviation, bit-layer configuration, frame length, CRC, and all recovered application payload bytes. The main weaknesses are CFO estimation, sync-word overextension, incomplete frame coverage, and the mismatch between the reported message count and the actual message array.
