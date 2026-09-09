# WAV-01 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `wav_01.wav` |
| Ground truth | `wav_01_truth.json` |
| Analyzer output | `analysis.json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | WAV |
| Modulation | QPSK |
| Sample rate | 1,000,000 Hz |
| Samples per symbol | 8.0 |
| Symbol rate | 125,000 Bd |
| Normalized symbol rate | 0.125 |
| SNR | 25.0 dB |
| CFO | +0.005 normalized |
| Roll-off | 0.35 |
| FEC | None |
| Interleaver | None |
| Scrambler | None |
| Frame length | 96 bits |
| Sync word | `eb90` |
| CRC | CRC-16-CCITT-FALSE |
| Payload length | 6 bytes/frame |

---

## Ground Truth vs Analyzer

| Parameter | Ground Truth | Analyzer Result | Error / Difference | Verdict |
|---|---:|---:|---:|---|
| Format | WAV | WAV | — | ✅ Correct |
| Sample rate | 1,000,000 Hz | 1,000,000 Hz | 0 | ✅ Correct |
| Modulation | QPSK | QPSK | — | ✅ Correct |
| Modulation confidence | — | 0.50 | QPSK/OQPSK tied | ⚠️ Low confidence |
| Samples/symbol | 8.0 | 8.0 | 0 | ✅ Correct |
| Symbol rate | 125,000 Bd | 125,000 Bd | 0% | ✅ Correct |
| Normalized symbol rate | 0.125 | 0.125 | 0 | ✅ Correct |
| CFO | +0.005 | ≈ +0.004996 combined | 0.0767% | ✅ Correct |
| SNR | 25.0 dB | 32.8 dB | +7.8 dB | ❌ Mismatch |
| Roll-off / excess BW* | 0.35 | 0.16 | −0.19 / 54.3% | ❌ Mismatch |
| FEC | None | None | — | ✅ Correct |
| Interleaver | None | None | — | ✅ Correct |
| Scrambler | None | None | — | ✅ Correct |
| Frame length | 96 bits | 96 bits | 0 | ✅ Correct |
| Sync word | `eb90` | `eb9000` | Extra `00` byte | ⚠️ Partial |
| CRC | CRC-16-CCITT-FALSE | CRC-16-CCITT-FALSE | — | ✅ Correct |
| Initial phase offset | 0.4 rad | No direct equivalent | — | ⚪ Not verifiable |

\*Scored only if analyzer `excess_bandwidth` is intended to represent the generator's pulse-shaping roll-off.

---

## CFO Evaluation

The analyzer channelized the signal first, so the reported residual CFO must be combined with the selected channel center.

| Quantity | Value |
|---|---:|
| Ground-truth CFO | +0.005 |
| Selected channel center | +0.00537109375 |
| Residual analyzer CFO | −0.00037493025 |
| Reconstructed CFO | +0.00499616350 |
| Absolute error | 0.00000383650 |
| Relative error | **0.0767%** |
| Verdict | ✅ Correct |

---

## SNR Evaluation

| Source | SNR |
|---|---:|
| Ground truth | 25.0 dB |
| Final analyzer parameter | 32.8 dB |
| Pipeline-trace S4 summary | 9.7 dB |

| Metric | Result |
|---|---:|
| Error using final analyzer value | +7.8 dB |
| Relative numerical difference | 31.2% |
| Internal consistency | ❌ Inconsistent |
| Verdict | ❌ Needs investigation |

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
| Raw recovered payload records | 66 |
| Correct recovered payloads | 66/66 |
| Raw payload accuracy | **100%** |
| Raw payload coverage | **82.5%** |
| CRC-validated frames | 64 |
| Correct CRC-validated payloads | 64/64 |
| CRC-validated payload accuracy | **100%** |
| CRC-validated frame coverage | **80%** |

### Payload Verdict

| Aspect | Verdict |
|---|---|
| Correctness of recovered payload bytes | ✅ Excellent |
| Coverage of all generated frames | ⚠️ Incomplete |
| Evidence quality | ✅ CRC validated |

---

## Internal Reporting Issues

| Issue | Observed Values | Verdict |
|---|---|---|
| SNR mismatch inside report | `32.8 dB` final vs `9.7 dB` S4 summary | ❌ Inconsistent |
| Message count mismatch | `n_messages = 66` but message array contains 64 entries | ❌ Inconsistent |
| Sync word length | Truth `eb90`, analyzer `eb9000` | ⚠️ Overestimated |

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Headline checks evaluated | 14 |
| Fully correct / strong | 11 |
| Partial | 1 |
| Material mismatches | 2 |
| Full-correct headline rate | **78.6%** |
| Symbol-rate error | **0%** |
| CFO relative error | **0.0767%** |
| SNR error | **+7.8 dB** |
| Roll-off / excess-BW error | **54.3%** |
| Recovered payload accuracy | **100%** |
| Raw payload coverage | **82.5%** |
| CRC-validated frame coverage | **80%** |
| CRC pass rate on tested frames | **100%** |

---

## Final Verdict

| Area | Result |
|---|---|
| Modulation identification | ✅ Correct |
| Demodulation | ✅ GOOD |
| Symbol-rate estimation | ✅ Exact |
| CFO estimation | ✅ Very accurate |
| FEC / interleaver / scrambler | ✅ Correct |
| Framing | ✅ Mostly correct |
| CRC | ✅ Correct |
| Payload recovery | ✅ 100% accurate on recovered frames |
| SNR estimation | ❌ Inaccurate / inconsistent |
| Roll-off estimation | ❌ Weak |
| Report consistency | ⚠️ Needs fixes |
| Overall | **PASS WITH ISSUES** |

The analyzer successfully identifies and decodes the QPSK signal, with excellent CFO, symbol-rate, CRC, and payload correctness. The main weaknesses in this test are SNR estimation, apparent roll-off estimation, incomplete frame coverage, sync-word overextension, and internal report inconsistencies.
