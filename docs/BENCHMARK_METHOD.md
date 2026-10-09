# How the numbers are made

Every number in the README is rendered by `scripts/render_readme.py` from
a `results/*.json` file that a script wrote. Each file starts with a
`provenance` block: UTC time, command, git commit (with a dirty flag),
CPU, platform, Python / numpy / scipy versions and a data label
(`synthetic` or `off-air`).

## Synthetic data

`dhwani.synth.WaveformFactory` builds frames (sync 0xEB90, counter,
constant byte, 6-byte payload, CRC-16/CCITT-FALSE), applies FEC,
interleaver and scrambler, modulates with RRC pulse shaping (rolloff
0.35), then adds CFO, phase offset and AWGN. Seeds are fixed and recorded
in each result's `settings`.

## `results/benchmark.json` (`scripts/benchmark.py`)

Wall-clock seconds of single calls on one machine: wideband detection
(2M samples), symbol-rate estimation, demodulation of about 100k
symbols, a GF(2) rank scan (L = 2..512, 50k bits), FEC identification,
one end-to-end analysis, and with `--big` loading a 1 GB IQ file to its
first waterfall. CVNet is off. The targets column holds the original
design budgets for comparison; it is not a pass/fail gate.

## `results/validation_report.json` (`scripts/validate_matrix.py`)

Every supported modulation at one or two SNRs (`MATRIX` in the script),
8 samples/symbol, 60 frames, CFO 0.004 and phase 0.3 for digital cases,
CVNet off.
Stress axes at the higher SNR (classification + demodulation only): every
digital modulation again at CFO 0, and the FSK family at 4 samples/symbol;
these rows are labelled "(CFO 0)" / "(4 sps)". Per case:

- blind S5 classification, top-1 and top-2 (QPSK/OQPSK and 2FSK/GMSK
  count as equivalent both ways, OOK->BPSK one way; the confusion matrix
  keeps the raw prediction);
- demodulation with the true modulation: status, EVM and a mid-window
  BER aligned against the transmitted bits;
- at the higher SNR, the full blind chain with the modulation pinned:
  frame found, CRC pass, payload trust, and whether the payload is a
  contiguous substring of what was sent.

Verdict: PASS = top-1 correct, BER < 2% and demodulation GOOD; DEGRADED
= BER < 2% but top-1 missed or demodulation DEGRADED; FAIL otherwise
(`FAIL(gated)` when the S6 gate refused the bits).

## `results/bitlayer_report.json` (`scripts/validate_bitlayer.py`)

QPSK with the modulation pinned, PN9 whitening, 80 frames, at 12 and
20 dB, for 5 interleavers (none, block 8x16, helical 8x16 step 3,
convolutional 4x8, pseudo-random period 128) x 5 FEC families (none,
conv K=7, RS(255,223), LDPC(256,128), RS+conv concatenated).
Identification is correct when kind and parameters match (pseudo-random:
the period). `payload_ber` is the bit error rate of the recovered
payload bytes at the best alignment with the transmitted payloads (1.0
when nothing was recovered); `frames_recovered` is the fraction of
payloads recovered.

## `results/offair_report.json` (`scripts/validate_offair.py`)

Licensed real recordings listed in a manifest with the parameters known
for them; per recording, which expected parameters were reproduced.
Labelled off-air. Nothing is reported until such recordings are added.

## What is not measured

PR curves (only S5 confidence and CFAR SNR are scored detectors), GUI
feature visibility beyond `tests/test_gui_smoke.py`, and anything off-air.
Analog cases are demodulated with the true modulation supplied, so they
PASS only when blind classification also named them; otherwise they are
DEGRADED with the reason stated.
