# WAV-05 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `wav_05.wav` |
| Ground truth | `wav_05_truth.json` |
| Analyzer output | `analysis(5).json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | WAV |
| Modulation | FM |
| Signal type | Analog |
| Sample rate | 480,000 Hz |
| SNR | 22.0 dB |
| CFO | +0.005 normalized |
| Phase offset | 0.4 rad |
| Digital framing | None |
| FEC | None |
| Interleaver | None |
| Scrambler | None |
| Payloads | None |

---

## Ground Truth vs Analyzer

| Parameter | Ground Truth | Analyzer Result | Error / Difference | Verdict |
|---|---:|---:|---:|---|
| Format | WAV | WAV | — | ✅ Correct |
| Input sample rate | 480,000 Hz | 480,000 Hz | 0 | ✅ Correct |
| Modulation | FM | UNKNOWN | — | ❌ Failed |
| Modulation confidence | — | 0.0 | — | ❌ Failed |
| SNR | 22.0 dB | 23.2 dB | +1.2 dB | ⚠️ Acceptable / moderate error |
| Selected-channel SNR | 22.0 dB | 32.9 dB | +10.9 dB | ❌ Poor |
| Coarse channel center | +0.005 | +0.006836 | 36.72% | ❌ Inaccurate |
| Residual CFO estimate | ≈ 0 after channelisation | +0.009719 | large | ❌ Inaccurate |
| Combined CFO | +0.005 | +0.016555 | 231.09% error | ❌ Failed |
| Digital symbol rate | Not meaningful for analog FM | 9,173.7 Bd | — | ⚠️ Should not be treated as validated |
| Samples/symbol | Not meaningful for analog FM | 40.46 | — | ⚠️ Should not be treated as validated |
| Demodulation | FM demod expected | Not attempted | — | ❌ Failed |
| FEC | Not applicable | Not evaluated | — | ✅ Correctly skipped downstream |
| Interleaver | Not applicable | Not evaluated | — | ✅ Correctly skipped downstream |
| Scrambler | Not applicable | Not evaluated | — | ✅ Correctly skipped downstream |
| Framing / CRC | Not applicable | Not evaluated | — | ✅ Correctly skipped downstream |
| Payload recovery | Not applicable | Not evaluated | — | ✅ Correctly skipped downstream |

---

## Modulation Classification

| Metric | Result | Verdict |
|---|---:|---|
| Ground-truth modulation | FM | — |
| Analyzer prediction | UNKNOWN | ❌ |
| Confidence | 0.0 | ❌ |
| Alternative reported internally | 8ASK at 0.0 confidence | ❌ |
| In-distribution flag | True | ⚠️ Contradictory with UNKNOWN outcome |
| Constant-envelope constraint | Applied | ✅ Reasonable observation |

---

## Parameter Estimation

| Metric | Ground Truth | Analyzer | Error | Verdict |
|---|---:|---:|---:|---|
| SNR | 22.0 dB | 23.2 dB | +1.2 dB | ⚠️ |
| Selected-segment SNR | 22.0 dB | 32.9 dB | +10.9 dB | ❌ |
| Channel center | +0.005000 | +0.006836 | 36.72% | ❌ |
| Combined CFO | +0.005000 | +0.016555 | 231.09% | ❌ |
| Symbol-rate estimate | N/A for analog FM | 9,173.7 Bd | — | ⚠️ Not meaningful |
| Samples/symbol estimate | N/A for analog FM | 40.46 | — | ⚠️ Not meaningful |

---

## Pipeline Outcome

| Stage | Result |
|---|---|
| S0 Ingestion | ✅ Successful |
| S1 Conditioning | ✅ Successful |
| S2 Signal detection | ✅ Successful |
| S3 Channelisation | ✅ Successful |
| S4 Parameter estimation | ⚠️ Mixed |
| S5 Modulation classification | ❌ UNKNOWN instead of FM |
| S6 Demodulation | ❌ Skipped |
| S7 Ambiguity / descrambling | ✅ Correctly irrelevant after analog classification failure |
| S8 De-interleaving | ✅ Not applicable |
| S9 FEC identification | ✅ Not applicable |
| S10 Framing / CRC | ✅ Not applicable |
| S11 Payload intelligence | ✅ Not applicable |
| S12 Reporting | ⚠️ Export path stopped after classification failure |

---

## Analog-Signal Applicability

| Digital Stage | Expected for FM? | Analyzer Behavior | Verdict |
|---|---|---|---|
| Scrambler detection | No | Skipped | ✅ Correct |
| Interleaver detection | No | Skipped | ✅ Correct |
| FEC detection | No | Skipped | ✅ Correct |
| Frame detection | No | Skipped | ✅ Correct |
| CRC detection | No | Skipped | ✅ Correct |
| Payload recovery | No | Skipped | ✅ Correct |

The truth explicitly marks this as an analog transmission with no digital framing. Therefore downstream digital stages must not be scored as failures.

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Format detection | ✅ Correct |
| Sample-rate detection | ✅ Correct |
| FM classification | **Failed** |
| Modulation confidence | **0.0** |
| Final SNR error | **+1.2 dB** |
| Selected-segment SNR error | **+10.9 dB** |
| Coarse center-frequency error | **36.72%** |
| Combined CFO error | **231.09%** |
| FM demodulation | **Not attempted** |
| Digital downstream stages | **Correctly not applicable** |

---

## Final Verdict

| Area | Result |
|---|---|
| Format detection | ✅ Correct |
| Signal detection | ✅ Successful |
| SNR estimation | ⚠️ Reasonable final estimate, inconsistent with segment estimate |
| CFO estimation | ❌ Poor |
| FM identification | ❌ Failed |
| FM demodulation | ❌ Not reached |
| Digital FEC / framing / payload | ✅ Correctly not applicable |
| **Overall** | **FAIL: MODULATION CLASSIFICATION** |

The analyzer successfully ingests and detects the FM recording, but fails to identify the signal as FM and instead returns `UNKNOWN` with zero confidence. This causes the pipeline to stop before FM demodulation. Since the ground truth explicitly defines the recording as analog FM with no digital framing, FEC, CRC, and payload stages are not applicable and should not be counted as failures.
