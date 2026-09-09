# WAV-02 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `wav_02.wav` |
| Ground truth | `wav_02_truth.json` |
| Analyzer output | `analysis(1).json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | WAV |
| Modulation | BPSK |
| Sample rate | 2,000,000 Hz |
| Samples per symbol | 8.0 |
| Symbol rate | 250,000 Bd |
| Normalized symbol rate | 0.125 |
| SNR | 18.0 dB |
| CFO | +0.005 normalized |
| Roll-off | 0.35 |
| FEC | Convolutional, K=7, rate 1/2, 171/133 octal |
| Interleaver | Block 8×16 |
| Scrambler | PN9-CC1101 |
| Frame length | 96 bits |
| Sync word | `eb90` |
| CRC | CRC-16-CCITT-FALSE |
| Payload length | 6 bytes/frame |

---

## Ground Truth vs Analyzer

| Parameter | Ground Truth | Analyzer Result | Error / Difference | Verdict |
|---|---:|---:|---:|---|
| Format | WAV | WAV | — | ✅ Correct |
| Sample rate | 2,000,000 Hz | 2,000,000 Hz | 0 | ✅ Correct |
| Modulation | BPSK | BPSK | — | ✅ Correct |
| Modulation confidence | — | 0.50 | BPSK/OOK tied | ⚠️ Low confidence |
| Samples/symbol | 8.0 | 8.0 | 0 | ✅ Correct |
| Symbol rate | 250,000 Bd | 250,000 Bd | 0% | ✅ Correct |
| Normalized symbol rate | 0.125 | 0.125 | 0 | ✅ Correct |
| CFO | +0.005 | +0.005000523 combined | 0.0105% | ✅ Correct |
| SNR | 18.0 dB | 26.7 dB | +8.7 dB | ❌ Mismatch |
| Roll-off / excess BW* | 0.35 | 0.17 | -0.18 / 51.4% | ❌ Mismatch |
| FEC family | Convolutional | convolutional | — | ✅ Correct |
| FEC K | 7 | 7 | 0 | ✅ Correct |
| FEC generators | 171/133 octal | 0o171/0o133 | — | ✅ Correct |
| FEC rate | 1/2 | 1/2 | — | ✅ Correct |
| Interleaver | Block 8×16 | 8×16 block | — | ✅ Correct |
| Scrambler | PN9-CC1101 | PN9-CC1101 | — | ✅ Correct |
| Scrambler polynomial | 545 | 545 | 0 | ✅ Correct |
| Scrambler degree | 9 | 9 | 0 | ✅ Correct |
| Scrambler seed/state | 511 at generator start | 252 at recovered stream | phase-aligned state differs | ⚪ Not directly comparable |
| Frame length | 96 bits | 96 bits | 0 | ✅ Correct |
| Sync word | `eb90` | `eb9000` | Extra `00` byte | ⚠️ Partial |
| CRC | CRC-16-CCITT-FALSE | CRC-16-CCITT-FALSE | — | ✅ Correct |
| Initial phase offset | 0.4 rad | No direct equivalent | — | ⚪ Not verifiable |

\*Only score roll-off against `excess_bandwidth` if those fields are intended to represent the same quantity.

---

## CFO Evaluation

| Quantity | Value |
|---|---:|
| Ground-truth CFO | +0.005000000 |
| Selected channel center | +0.005004883 |
| Residual analyzer CFO | -0.000004360 |
| Reconstructed CFO | +0.005000523 |
| Absolute error | 0.000000523 |
| Relative error | **0.0105%** |
| Verdict | ✅ Correct |

---

## SNR Evaluation

| Source | SNR |
|---|---:|
| Ground truth | 18.0 dB |
| Selected-segment estimate | 26.48 dB |
| Final analyzer parameter | 26.7 dB |
| Pipeline S4 summary | `Rs 0.125 (norm), SNR 5.4 dB, OBW 0.14599609375` |

| Metric | Result |
|---|---:|
| Final-value error | +8.7 dB |
| Relative numerical difference | 48.3% |
| Internal consistency | ❌ Multiple conflicting estimates |
| Verdict | ❌ Needs investigation |

---

## Scrambler, Interleaver and FEC

| Property | Ground Truth | Analyzer | Verdict |
|---|---|---|---|
| Scrambler family | Known whitening | known_whitening | ✅ Correct |
| Scrambler name | PN9-CC1101 | PN9-CC1101 | ✅ Correct |
| Polynomial | 545 / `0x221` | 545 / `0x221` | ✅ Correct |
| Degree | 9 | 9 | ✅ Correct |
| Interleaver type | Block | block | ✅ Correct |
| Interleaver dimensions | 8×16 | 8×16 | ✅ Correct |
| Interleaver period | 128 bits | 128 bits | ✅ Correct |
| FEC family | Convolutional | convolutional | ✅ Correct |
| Constraint length K | 7 | 7 | ✅ Correct |
| Generators | 171/133 octal | 0o171/0o133 | ✅ Correct |
| Code rate | 1/2 | 1/2 | ✅ Correct |
| Syndrome-zero rate | — | 1.000 | ✅ Strong evidence |
| Pre-FEC BER | — | 0.0434% | ✅ Very low |

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
| Generated application payloads | 80 |
| Raw recovered payload records | 72 |
| Correct raw recovered payloads | 72/72 |
| Raw payload accuracy | **100.0%** |
| Raw payload coverage | **90.0%** |
| CRC-validated/reconstructed messages present | 64 |
| Correct reconstructed message payloads | 64/64 |
| Reconstructed payload accuracy | **100.0%** |
| CRC-valid frames reported | 64 |
| CRC-validated frame coverage | **80.0%** |

| Aspect | Verdict |
|---|---|
| Correctness of recovered application bytes | ✅ 100% |
| Coverage of all generated payloads | ⚠️ Incomplete |
| CRC evidence | ✅ 100% pass on validated frames |

---

## Internal Reporting Issues

| Issue | Observed Values | Verdict |
|---|---|---|
| SNR values disagree | Truth 18.0 dB; selected 26.48 dB; final 26.7 dB; S4 trace says 5.4 dB | ❌ Inconsistent |
| Message count mismatch | `n_messages = 72` but message array contains 64 entries | ❌ Inconsistent |
| Sync word length | Truth `eb90`, analyzer `eb9000` | ⚠️ Overestimated |
| Scrambler seed/state | Truth 511, analyzer 252 | ⚪ Requires common bit-origin before scoring |

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Symbol-rate error | **0%** |
| CFO relative error | **0.0105%** |
| SNR error | **+8.7 dB** |
| Roll-off / excess-BW error | **51.4%** |
| Pre-FEC BER | **0.0434%** |
| Raw recovered payload accuracy | **100.0%** |
| Raw payload coverage | **90.0%** |
| Reconstructed payload accuracy | **100.0%** |
| CRC-validated frame coverage | **80.0%** |
| CRC pass rate on tested frames | **100.0%** |

---

## Final Verdict

| Area | Result |
|---|---|
| Modulation identification | ✅ Correct, low confidence |
| Demodulation | ✅ GOOD |
| Symbol-rate estimation | ✅ Exact |
| CFO estimation | ✅ Extremely accurate |
| Scrambler identification | ✅ Correct |
| Interleaver identification | ✅ Correct |
| FEC identification | ✅ Correct |
| Framing | ✅ Mostly correct |
| CRC | ✅ Correct |
| Payload correctness | ✅ 100% on recovered records |
| Payload coverage | ⚠️ Incomplete |
| SNR estimation | ❌ Inaccurate / inconsistent |
| Roll-off estimation | ❌ Weak |
| Report consistency | ⚠️ Needs fixes |
| **Overall** | **PASS WITH ISSUES** |

The analyzer successfully recovers the substantially more complex BPSK chain, including PN9 whitening, 8×16 block interleaving, K=7 convolutional FEC, framing, CRC, and application payloads. Its main weaknesses remain SNR/roll-off estimation, sync-word overextension, incomplete frame coverage, and internal reporting consistency.
