# WAV-03 Evaluation Report

## Test Files

| Item | File |
|---|---|
| Input signal | `wav_03.wav` |
| Ground truth | `wav_03_truth.json` |
| Analyzer output | `analysis(2).json` |

---

## Test Configuration

| Parameter | Ground Truth |
|---|---:|
| Format | WAV |
| Modulation | GMSK |
| Sample rate | 1,500,000 Hz |
| Samples per symbol | 8.0 |
| Symbol rate | 187,500 Bd |
| Normalized symbol rate | 0.125 |
| SNR | 14.0 dB |
| CFO | +0.005 normalized |
| FSK/GMSK deviation | 0.0625 normalized |
| FEC | LDPC (128/256, seed 1) |
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
| Sample rate | 1,500,000 Hz | 1,500,000 Hz | 0 | ✅ Correct |
| Modulation | GMSK | GMSK | — | ✅ Correct |
| Modulation confidence | — | 1.00 | — | ✅ Strong |
| SNR | 14.0 dB | 14.0 dB | 0 dB | ✅ Exact |
| Parameter-stage samples/symbol | 8.0 | 30.872 | large error | ❌ Wrong |
| Parameter-stage symbol rate | 187,500 Bd | 48,588 Bd | 74.1% | ❌ Wrong |
| Demodulator samples/symbol | 8.0 | 8.0 | 0 | ✅ Correct |
| Demodulator normalized symbol rate | 0.125 | 0.125 | 0 | ✅ Correct |
| Selected-channel center | +0.005 expected | +0.005004883 | 0.098% | ✅ Very accurate |
| Final parameter CFO after combining | +0.005 | -0.026249477 | 625.0% | ❌ Wrong |
| FSK/GMSK deviation | 0.062500 | 0.006621 | 89.4% | ❌ Wrong |
| FEC | LDPC | Not recovered | — | ❌ Failed |
| Interleaver | None | Not evaluated | — | ⚠️ Downstream not reached |
| Scrambler | None | Not evaluated | — | ⚠️ Downstream not reached |
| Frame length | 96 bits | Not recovered | — | ❌ Failed |
| Sync word | `eb90` | Not recovered | — | ❌ Failed |
| CRC | CRC-16-CCITT-FALSE | Not recovered | — | ❌ Failed |
| Payload | 80 known payloads | No payload recovered | 0% coverage | ❌ Failed |

---

## Modulation and Demodulation

| Metric | Result | Verdict |
|---|---:|---|
| Modulation prediction | GMSK | ✅ Correct |
| Confidence | 1.0 | ✅ Strong |
| Demodulation status | GOOD | ✅ |
| Carrier lock | True | ✅ |
| Timing lock | True | ✅ |
| Demodulated symbols | 14,336 | — |
| Demodulated bits | 14,336 | — |
| Demodulator samples/symbol | 8.0 | ✅ Exact |
| Squaring-line Rs (normalized) | 0.125 | ✅ Exact |
| Squaring-line quality | 240.1 | ✅ Strong |
| Discriminator margin | 0.725 | ✅ |

---

## Parameter Estimation

| Metric | Ground Truth | Analyzer | Error | Verdict |
|---|---:|---:|---:|---|
| SNR | 14.0 dB | 14.0 dB | 0 dB | ✅ Exact |
| Symbol rate | 187,500 Bd | 48,588 Bd | 74.1% | ❌ |
| Samples/symbol | 8.0 | 30.872 | 22.872 | ❌ |
| Channel center / coarse CFO | +0.005000 | +0.005005 | 0.098% | ✅ |
| Residual CFO estimate | ≈ 0 after channelization | -0.031254 | large | ❌ |
| Combined CFO | +0.005000 | -0.026249 | 625.0% | ❌ |
| GMSK deviation | 0.062500 | 0.006621 | 89.4% | ❌ |

### Internal Rate Inconsistency

| Stage | Symbol-rate result | Verdict |
|---|---:|---|
| Ground truth | 0.125 norm / 187,500 Bd | — |
| S4 parameter estimator | 0.032392 norm / 48,588 Bd | ❌ Wrong |
| S6 GMSK demodulator | 0.125 norm / 8 samples/symbol | ✅ Correct |

---

## Bit-Layer / FEC / Framing

| Stage | Ground Truth | Analyzer Result | Verdict |
|---|---|---|---|
| Candidate bit streams after S7 | Should allow downstream processing | 0 streams | ❌ |
| Scrambler | None | `null` | ⚠️ Not evaluated |
| Interleaver | None | `null` | ⚠️ Not evaluated |
| FEC | LDPC 128/256 | `null` | ❌ Not recovered |
| Frame structure | 96-bit frames | `null` | ❌ Not recovered |
| Sync | `eb90` | `null` | ❌ Not recovered |
| CRC | CRC-16-CCITT-FALSE | `null` | ❌ Not recovered |
| Payload | 80 generated payloads | `null` | ❌ Not recovered |

---

## Payload Recovery

| Metric | Result |
|---|---:|
| Generated payloads | 80 |
| Recovered payloads | 0 |
| Payload coverage | **0%** |
| CRC-validated frames | 0 |
| CRC-validated frame coverage | **0%** |
| Payload-byte accuracy | Not measurable because no payload was recovered |

---

## Pipeline Outcome

| Stage | Result |
|---|---|
| S0 Ingestion | ✅ Successful |
| S1 Conditioning | ✅ Successful |
| S2 Detection | ✅ Successful |
| S3 Channelisation | ✅ Successful |
| S4 Parameter estimation | ⚠️ Mixed: SNR correct, rate/CFO/deviation wrong |
| S5 Modulation classification | ✅ GMSK, confidence 1.0 |
| S6 Demodulation | ✅ `GOOD` |
| S7 Ambiguity / bit-stream fan-out | ❌ 0 candidate streams |
| S8 De-interleaving | ❌ No hypothesis |
| S9 FEC identification | ❌ No hypothesis; LDPC missed |
| S10 Framing / CRC | ❌ No frame structure |
| S11 Payload intelligence | ❌ Skipped; no payload |

---

## Summary Metrics

| Metric | Result |
|---|---:|
| Modulation accuracy | ✅ Correct |
| Modulation confidence | **1.0** |
| SNR error | **0 dB** |
| S4 symbol-rate error | **74.1%** |
| S6 demodulator rate error | **0%** |
| Coarse channel-center error | **0.098%** |
| Combined reported CFO error | **625.0%** |
| GMSK deviation error | **89.4%** |
| LDPC recovery | **Failed** |
| Frame recovery | **0%** |
| Payload coverage | **0%** |

---

## Final Verdict

| Area | Result |
|---|---|
| Format detection | ✅ Correct |
| GMSK identification | ✅ Excellent |
| SNR estimation | ✅ Exact |
| Demodulation lock | ✅ GOOD |
| S4 symbol-rate reporting | ❌ Incorrect |
| S6 symbol-rate recovery | ✅ Correct |
| CFO reporting | ❌ Inconsistent / incorrect after channelization |
| GMSK deviation estimation | ❌ Inaccurate |
| LDPC identification | ❌ Failed |
| Framing / CRC | ❌ Failed |
| Payload recovery | ❌ Failed |
| **Overall** | **PARTIAL / DOWNSTREAM FAILURE** |

The analyzer correctly identifies and demodulates the GMSK waveform, including an exact SNR estimate and correct symbol timing inside the demodulator. However, S4 reports the wrong symbol rate and residual CFO, and the pipeline produces zero candidate bit streams after demodulation. As a result, the expected LDPC, framing, CRC, and payload stages are never recovered.
