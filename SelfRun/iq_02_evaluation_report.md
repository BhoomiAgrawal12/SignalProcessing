# IQ-02 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `iq_02.iq` |
| Ground truth | `iq_02_truth.json` |
| Analyzer output | `analysis(7).json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | Raw IQ |
| Modulation | 16QAM |
| Sample rate | 1,800,000 Hz |
| Samples per symbol | 8.0 |
| Symbol rate | 225,000 Bd |
| Normalized symbol rate | 0.125 |
| SNR | 28.0 dB |
| CFO | +0.005 normalized |
| Roll-off | 0.35 |
| FEC | Convolutional, K=7, rate 1/2, 171/133 octal |
| Interleaver | Block 8×16 |
| Scrambler | PN9-CC1101 |
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
| Format confidence | — | 0.226 | low | ⚠️ Low confidence |
| Sample rate | 1,800,000 Hz | 1,800,000 Hz | 0 | ✅ Correct* |
| Modulation | 16QAM | 64QAM | Misclassified | ❌ Wrong |
| Confidence | — | 0.400 | true 16QAM only 0.021 | ❌ Weak |
| Samples/symbol | 8.0 | 8.0 | 0.0000% | ✅ Exact |
| Symbol rate | 225,000 Bd | 225,000 Bd | 0.0000% | ✅ Exact |
| SNR | 28.0 dB | 18.5 dB | -9.5 dB | ❌ Poor |
| Combined CFO | +0.005000 | +0.005005 | 0.0977% | ✅ Excellent |
| Roll-off / excess BW** | 0.35 | 0.16 | 54.3% | ❌ Mismatch |
| Scrambler | PN9-CC1101 | None | — | ❌ Missed |
| Interleaver | Block 8×16 | None | — | ❌ Missed |
| FEC | Convolutional K=7, 171/133 | None | — | ❌ Missed |
| Frame | 96-bit, `eb90` | None | — | ❌ Not recovered |
| CRC | CRC-16-CCITT-FALSE | None | — | ❌ Not recovered |
| Payload | 80 generated payloads | None | 0% coverage | ❌ Failed |

\*Raw IQ sample rate was supplied during analysis.
\**Only directly comparable if `excess_bandwidth` is intended to estimate generator roll-off.

---

## Modulation Classification

| Candidate | Analyzer Confidence |
|---|---:|
| 64QAM | **0.400** |
| 256QAM | 0.291 |
| 32QAM | 0.188 |
| 128QAM | 0.096 |
| **16QAM (truth)** | **0.021** |

| Metric | Result |
|---|---|
| Ground truth | 16QAM |
| Analyzer prediction | 64QAM |
| Verdict | ❌ Misclassification |

---

## Demodulation

| Metric | Result | Verdict |
|---|---:|---|
| Demodulation attempted as | 64QAM | ❌ Wrong model |
| Status | DEGRADED | ⚠️ |
| EVM | 11.9% | ⚠️ Marginal |
| Carrier lock | True | ✅ |
| Timing lock | True | ✅ |
| Demodulated symbols | 3,225 | — |
| Demodulated bits | 19,350 | — |

The analyzer explicitly warns that downstream results are speculative because 64QAM EVM is marginal.

---

## Parameter Estimation

| Metric | Ground Truth | Analyzer | Error | Verdict |
|---|---:|---:|---:|---|
| Symbol rate | 225,000 Bd | 225,000 Bd | 0.0000% | ✅ |
| Samples/symbol | 8.0 | 8.0 | 0.0000% | ✅ |
| CFO | +0.005000 | +0.005005 | 0.0977% | ✅ |
| SNR | 28.0 dB | 18.5 dB | -9.5 dB | ❌ |
| Selected-segment SNR | 28.0 dB | 36.3 dB | +8.3 dB | ❌ |
| Roll-off / excess BW | 0.35 | 0.16 | 54.3% | ❌ |

---

## Bit-Layer Recovery

| Stage | Ground Truth | Analyzer | Verdict |
|---|---|---|---|
| Scrambler | PN9-CC1101 | None | ❌ |
| Interleaver | Block 8×16 | None | ❌ |
| FEC | Convolutional K=7, 171/133, rate 1/2 | None | ❌ |
| Frame structure | 96 bits | None | ❌ |
| Sync | `eb90` | None | ❌ |
| CRC | CRC-16-CCITT-FALSE | None | ❌ |
| Payload | 80 payloads | None | ❌ |

---

## Pipeline Outcome

| Stage | Result |
|---|---|
| S0 Ingestion | ✅ Successful |
| S1 Conditioning | ✅ Successful |
| S2 Detection | ✅ Successful |
| S3 Channelisation | ✅ Successful |
| S4 Parameter estimation | ⚠️ Rate/CFO strong, SNR weak |
| S5 Modulation classification | ❌ 64QAM instead of 16QAM |
| S6 Demodulation | ⚠️ DEGRADED under wrong modulation |
| S7 Descrambling | ❌ PN9 missed |
| S8 De-interleaving | ❌ Block interleaver missed |
| S9 FEC identification | ❌ Convolutional code missed |
| S10 Framing / CRC | ❌ No frame structure |
| S11 Payload | ❌ No payload recovered |

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Raw-IQ ingestion | ✅ Correct |
| Modulation accuracy | **0% for this test** |
| Symbol-rate error | **0.0000%** |
| Samples/symbol error | **0.0000%** |
| CFO error | **0.0977%** |
| SNR error | **-9.5 dB** |
| Scrambler recovery | **Failed** |
| Interleaver recovery | **Failed** |
| FEC recovery | **Failed** |
| Frame recovery | **Failed** |
| Payload coverage | **0%** |

---

## Final Verdict

| Area | Result |
|---|---|
| Raw-IQ ingestion | ✅ Correct |
| Symbol-rate estimation | ✅ Exact |
| CFO estimation | ✅ Excellent |
| 16QAM classification | ❌ Failed |
| Demodulation | ⚠️ Degraded under wrong 64QAM model |
| PN9 recovery | ❌ Failed |
| Block interleaver recovery | ❌ Failed |
| Convolutional FEC recovery | ❌ Failed |
| Framing / CRC | ❌ Failed |
| Payload recovery | ❌ Failed |
| **Overall** | **FAIL: MODULATION MISCLASSIFICATION / DOWNSTREAM FAILURE** |

The analyzer estimates the raw-IQ sample rate input, symbol rate and CFO very accurately, but misclassifies the true 16QAM signal as 64QAM. Because demodulation proceeds using the wrong constellation, the PN9 scrambler, block interleaver, convolutional FEC, frame structure, CRC and payload are all missed. This is an end-to-end failure caused primarily by the modulation-classification stage.
