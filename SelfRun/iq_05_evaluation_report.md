# IQ-05 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `iq_05.iq` |
| Ground truth | `iq_05_truth.json` |
| Analyzer output | `analysis(10).json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | Raw IQ |
| Modulation | 16APSK |
| Sample rate | 2,000,000 Hz |
| Samples per symbol | 8.0 |
| Symbol rate | 250,000 Bd |
| Normalized symbol rate | 0.125 |
| SNR | 26.0 dB |
| CFO | +0.005 normalized |
| Roll-off | 0.35 |
| FEC | LDPC 128/256 |
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
| Format confidence | — | 0.224 | low | ⚠️ Low confidence |
| Sample rate | 2,000,000 Hz | 2,000,000 Hz | 0 | ✅ Correct* |
| Modulation | 16APSK | 128QAM | Misclassified | ❌ Wrong |
| Analyzer confidence | — | 0.362 | true 16APSK only 0.016 | ❌ Weak |
| Samples/symbol | 8.0 | 8.0 | 0.0000% | ✅ Exact |
| Symbol rate | 250,000 Bd | 250,000 Bd | 0.0000% | ✅ Exact |
| Final SNR | 26.0 dB | 4.6 dB | -21.4 dB | ❌ Severe error |
| Selected-segment SNR | 26.0 dB | 34.34 dB | +8.34 dB | ❌ Overestimated |
| Coarse channel center | +0.005000 | +0.005615 | 12.30% | ⚠️ Approximate |
| Combined S4 CFO | +0.005000 | -0.020717 | 514.3% | ❌ Wrong |
| Roll-off / excess BW** | 0.35 | 0.17 | 51.4% | ❌ Mismatch |
| LDPC | 128/256 | Not evaluated | — | ❌ Not recovered |
| Frame / CRC | 96-bit / CRC-16-CCITT-FALSE | Not evaluated | — | ❌ Not recovered |
| Payload | 80 generated | None | 0% coverage | ❌ Failed |

\*Raw IQ sample rate was supplied during analysis.
\**Only directly comparable if `excess_bandwidth` is intended to estimate generator roll-off.

---

## Modulation Classification

| Candidate | Analyzer Confidence |
|---|---:|
| 128QAM | **0.362** |
| 32QAM | 0.308 |
| 128APSK | 0.174 |
| 32APSK | 0.135 |
| **16APSK (truth)** | **0.016** |

| Metric | Result |
|---|---|
| Ground truth | 16APSK |
| Analyzer prediction | 128QAM |
| True modulation rank | 5th |
| Verdict | ❌ Major misclassification |

---

## Demodulation

| Metric | Result | Verdict |
|---|---:|---|
| Demodulation attempted as | 128QAM | ❌ Wrong modulation |
| Status | FAILED | ❌ |
| EVM | 9.5% | ⚠️ |
| Carrier lock | False | ❌ |
| Timing lock | True | ✅ |
| Symbols examined | 3,225 | — |
| Bit stream released | No | ✅ Correctly withheld |

The S6 quality gate correctly stops the pipeline when carrier lock fails, preventing unreliable bits from being passed into LDPC/framing stages.

---

## Parameter Estimation

| Metric | Ground Truth | Analyzer | Error | Verdict |
|---|---:|---:|---:|---|
| Symbol rate | 250,000 Bd | 250,000 Bd | 0.0000% | ✅ |
| Samples/symbol | 8.0 | 8.0 | 0.0000% | ✅ |
| Final SNR | 26.0 dB | 4.6 dB | -21.4 dB | ❌ |
| Selected-segment SNR | 26.0 dB | 34.34 dB | +8.34 dB | ❌ |
| Selected center | +0.005000 | +0.005615 | 12.30% | ⚠️ |
| Combined S4 CFO | +0.005000 | -0.020717 | 514.3% | ❌ |
| Roll-off / excess BW | 0.35 | 0.17 | 51.4% | ❌ |

---

## Downstream Recovery

| Stage | Ground Truth | Analyzer | Verdict |
|---|---|---|---|
| Scrambler | None | Not evaluated | ⚠️ Pipeline stopped |
| Interleaver | None | Not evaluated | ⚠️ Pipeline stopped |
| FEC | LDPC 128/256 | Not evaluated | ❌ Not recovered |
| Frame structure | 96 bits | Not evaluated | ❌ Not recovered |
| Sync | `eb90` | Not evaluated | ❌ Not recovered |
| CRC | CRC-16-CCITT-FALSE | Not evaluated | ❌ Not recovered |
| Payload | 80 payloads | None | ❌ 0% coverage |

---

## Pipeline Outcome

| Stage | Result |
|---|---|
| S0 Ingestion | ✅ Successful |
| S1 Conditioning | ✅ Successful |
| S2 Detection | ✅ Successful |
| S3 Channelisation | ✅ Successful |
| S4 Parameter estimation | ⚠️ Rate exact, SNR/CFO poor |
| S5 Modulation classification | ❌ 128QAM instead of 16APSK |
| S6 Demodulation | ❌ FAILED, no carrier lock |
| S7 Descrambling | ⛔ Correctly skipped by quality gate |
| S8 De-interleaving | ⛔ Correctly skipped |
| S9 LDPC identification | ⛔ Correctly skipped, therefore LDPC not recovered |
| S10 Framing / CRC | ⛔ Correctly skipped |
| S11 Payload intelligence | ⛔ Correctly skipped |

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Raw-IQ ingestion | ✅ Correct |
| 16APSK classification | **Failed** |
| True-class confidence | **0.016** |
| Symbol-rate error | **0.0000%** |
| Samples/symbol error | **0.0000%** |
| Final SNR error | **-21.4 dB** |
| Combined S4 CFO error | **514.3%** |
| Carrier lock | **Failed** |
| LDPC recovery | **Not reached / failed** |
| Frame recovery | **0%** |
| Payload coverage | **0%** |
| S6 quality gate | ✅ Correctly blocked unreliable bits |

---

## Final Verdict

| Area | Result |
|---|---|
| Raw-IQ ingestion | ✅ Correct, low sniff confidence |
| Symbol-rate estimation | ✅ Exact |
| 16APSK identification | ❌ Failed |
| Demodulation | ❌ Failed under wrong 128QAM model |
| SNR estimation | ❌ Severe error |
| CFO estimation | ❌ Poor |
| LDPC recovery | ❌ Not reached |
| Framing / CRC | ❌ Not reached |
| Payload recovery | ❌ 0% |
| Quality-gate behavior | ✅ Correct |
| **Overall** | **FAIL: MODULATION MISCLASSIFICATION / DEMODULATION FAILURE** |

The analyzer correctly ingests the raw-IQ file and recovers the symbol rate exactly, but it misclassifies the true 16APSK waveform as 128QAM. Demodulation then fails to obtain carrier lock, and the S6 quality gate correctly withholds the unreliable bit stream. Consequently, the expected LDPC, framing, CRC and payload stages are never reached. The test also exposes severe SNR and CFO estimation errors.
