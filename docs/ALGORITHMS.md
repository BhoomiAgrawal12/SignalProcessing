# Algorithms

For each major algorithm: what it does, inputs/outputs, assumptions and
limitations.

## S0 raw-IQ format sniffing (`ingestion/sniffer.py`)
- Tests: file-size divisibility per dtype, value-histogram plausibility
  (uint8 centred near 127.5, int8/int16 near 0, float magnitudes sane),
  endianness both ways, spectral structure (max-median PSD crest: the
  correct interpretation of a real recording is band-limited; a wrong
  dtype reads as white noise), spectral symmetry (annotation for
  possibly-real data).
- Output: ranked candidates with confidence and a one-line explanation.
- Limitation: cannot distinguish complex64 from interleaved float32
  (identical bytes); complex64 is preferred by convention.

## S1 conditioning (`conditioning/conditioner.py`)
- DC removal (mean), clipping fraction at 98.5% of peak, dead-air
  fraction from 4096-sample frame powers, Gram-Schmidt I/Q
  orthogonalisation gated on measured gain/phase imbalance, image
  rejection measured before/after.

## S2 detection (`detection/cfar.py`)
- STFT waterfall with FFT size adapted so short recordings still give
  >= 16 time rows. Noise floor: median over time per bin, median-filtered
  over frequency, capped by the 25th percentile of the band (assumes
  >= 25% of the band is signal-free; strongly coloured floors are a
  known limitation). Threshold floor + margin (default 8 dB), binary
  closing/opening, connected components, per-box SNR.

## S4 parameter estimation (`params/estimators.py`)
- OBW: ITU-R SM.443 99% power bandwidth on a floor-subtracted Welch PSD.
- SNR: M2M4 moment estimator (assumes constant-modulus signal; QAM and
  oversampling bias it low - the demodulator's EVM is the better
  post-hoc estimate).
- CFO: strongest spectral line of x^M for M in {2,4,8}; hard limiter
  applied for constant-modulus signals only (it destroys the QAM x^4
  line).
- Symbol rate: time-smoothed cyclic periodograms of x(t)x*(t-d) summed
  over delays d in {0,1,2,4}; candidate peaks scored by prominence over a
  median-filtered background; OBW guard rejects low-frequency envelope
  spurs; harmonics of kept candidates suppressed.
- FSK: constant envelope gate (amplitude cv <= 0.22, measured margins:
  shaped PSK/QAM >= 0.27, FSK <= 0.13); adaptive smoothing window (2..32
  samples) chosen by histogram concentration; tone count from interior
  peaks of the widened histogram; symbol rate from the spectral line of
  |d inst_f/dt| with sub-harmonic preference.
- OFDM: cyclic-prefix autocorrelation requiring both strong correlation
  and a bursty (peaky) profile, sizes >= 128/CP >= 16; final OFDM verdict
  additionally requires a noise-like envelope and *absent* single-carrier
  cumulant structure.

## S5 classification (`modulation/`)
- Cumulant engine: features [|C20|, |C40|, |C42|, m63] computed on
  CFO-corrected, timing-recovered symbols; references computed exactly
  from the constellation tables with the same estimators (estimator bias
  cancels); temperature-scaled softmax over weighted distances, softened
  below 15 dB SNR.
- CVNet-RF: frames of 1024 raw IQ samples, per-channel z-score, softmax
  averaged over up to 32 frames; 24 RadioML classes mapped onto the
  supported set, out-of-distribution mass reported.
- Fusion: hard constraints first (FSK tones decide; constant envelope
  excludes QAM; OFDM only without cumulant structure), then a weighted
  vote; disagreement flagged. Measured: 42/42 correct on the synthetic
  matrix (6 seeds x 7 modulations, SNR 15-30 dB).

## S6 demodulation (`demod/receiver.py`)

Family dispatch: psk / oqpsk / qam / apsk / ask (linear chain), fsk /
gmsk (discriminator chain), analog (audio detectors). Every path ends in
the quality gate that sets `demodulation_status` from per-modulation EVM
gates, lock flags and the rotational-concentration metric; FAILED
demodulations never feed the bit layer.

Key design facts (each was verified against ground truth the hard way):
- Nearest-point EVM cannot detect a spinning dense-PSK constellation, so
  carrier lock is additionally evidenced by rotational concentration
  (|E[u^M]| for PSK, ring-gated for APSK, an off-grid-rotation geometric
  baseline for QAM).
- CFO is estimated in the symbol domain after timing (ISI-free samples);
  candidate spectral lines are arbitrated by nearest-constellation
  distance because frame-periodic data creates lines that masquerade as
  CFO. Sample-domain M-power pre-correction is only applied for PSK
  orders <= 8 and ASK.
- Dense PSK/APSK use slip-free block Viterbi&Viterbi feedforward carrier
  recovery; dense QAM (128/256) uses no tracking loop at all - a fine
  constant-phase grid search plus a very slow feedforward polish, because
  both DD-PLLs and per-block VV add decision noise there.
- APSK coset ambiguity: the VV ring anchor fixes phase modulo the
  dominant ring's own symmetry; the residual rotation is snapped against
  the full table on the non-dominant rings, and the enumeration uses the
  constellation's TRUE symmetry group computed from the table.
- OQPSK recovers the carrier first (x^4 line at 4fc), takes the symbol
  rate from the x^2 line pair symmetric about 2fc, and tries both stagger
  directions. GMSK takes rate and carrier from the classic x^2 squaring
  lines at 2fc +- Rs/2.
- Timing is feedforward Oerder&Meyr with cubic Lagrange interpolation.

### Legacy notes
- Timing: Oerder&Meyr feedforward estimator - the phase of the symbol-rate
  spectral tone of |x|^2 per block, unwrapped across blocks, then a
  weighted linear fit gives timing offset + clock drift; symbols are
  interpolated at the fitted instants. Deterministic; no feedback loop to
  diverge. Assumes near-constant clock offset over the burst.
- Carrier: decision-directed PLL (2nd order); 4th-power feedforward warm
  start for QPSK/QAM.
- Slicing: nearest constellation point; per-bit max-log LLRs from
  set-distance differences; EVM against the ideal table.
- FSK: smoothed instantaneous frequency, tone centres from the histogram,
  best sampling phase by tone-distance cost, LLRs from distance margins.

## S8 blind de-interleaving (`interleaving/detect.py`)
- Rank-deficiency profile over L (Sicot & Houcke): significance =
  deficiency minus the closed-form expected deficiency of a random
  matrix. Fundamental period = smallest significant L. Dense significance
  across the detectable range (L <= sqrt(2 n_bits)) means "plain linear
  code, no interleaver".
- Hypothesis tests: for each factorisation (r, c) of P (and helical step,
  and convolutional branch/delay), sweep the block-phase offset 0..P-1
  and score by fine-grained structure significance of the de-interleaved
  stream at small L. The correct (kind, params, offset) makes code
  structure reappear below P ("second, deeper dip").
- Unresolved period -> honest pseudo-random verdict with period only.

## S9 blind FEC identification (`fec/detect.py`)
- Convolutional (rate 1/2): dual-code test - the true generators satisfy
  g2(D)v1(D) + g1(D)v2(D) = 0. Sweep standard codes then exhaustive
  K <= 7 pairs; the *constancy* of the syndrome (all-0 or all-1) is the
  detection statistic, which makes the test invariant to the substream
  inversions produced by constellation rotation ambiguities. The
  inversion pattern is resolved at decode time by re-encoding the Viterbi
  output and measuring BER against the observed stream.
- Reed-Solomon: candidate table (CCSDS RS(255,223) first) swept over bit
  offset (0..m-1) and symbol offset (0..n-1), scored by the all-zero
  syndrome fraction; a quick pass on few codewords prunes, a confirmation
  pass on up to 64 codewords scores. Skipped when a confident
  convolutional hit exists or the stream is differential-decoded.
- Confidence metric everywhere: syndrome-zero rate (chance level ~0.5 for
  conv parity, ~2^-16t for RS).

## S7 scrambler handling (`scrambling/`)
- Key fact: an additive LFSR scrambler adds only a (degree+1)-dimensional
  affine overlay to GF(2) row spaces, so period detection works *through*
  the scrambler, and the correct whitener phase is the one that maximises
  rank deficiency at L = P. Library whiteners: PN9/CC1101, CCSDS, DVB,
  IEEE 802.11 (DVB's 2^15-1 period is truncated to 512 phases at
  interactive speed and reported as such).
- Berlekamp-Massey recovers short LFSRs directly from idle/preamble
  stretches; the connection polynomial is converted to the Fibonacci tap
  mask (reciprocal) before verification.

## S10 framing (`framing/frames.py`, `framing/crc.py`)
- Frame period: bit-packed autocorrelation plus column-stability scan
  with a small-sample bias correction (0.798/sqrt(rows)) and explicit
  fundamental-period preference (divisor testing).
- Field map: per-column Shannon entropy across stacked frames; contiguous
  segments classified sync/header/payload by entropy level; sync word =
  longest near-constant run, and the stream is re-stacked so frames start
  at the sync word before CRC hunting.
- CRC: parametric engine (width, poly, init, refin, refout, xorout)
  verified against published check values; staged hunt over the candidate
  table and plausible start bytes, CRC anchored at frame end.
- Payload: bytes between sync end and CRC start; byte-entropy and
  printability decide the "likely encrypted" flag (entropy > 0.95 b/b).
