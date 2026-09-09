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
- The band is segmented at SEVERAL analysis resolutions (Welch segment
  lengths 256 to 32768) against SEVERAL noise-floor models, and the
  segmentation that produces the most compact, highest-contrast box wins.
  A single resolution with a single floor model cannot both resolve a
  narrowband carrier inside a 48 kHz audio capture and keep an ordinary
  wideband burst in one piece.
- Noise-floor models, all tried, none privileged:
  - running low quantile over frequency at three window widths
    (band/32, band/8, band/2) - tracks an arbitrarily shaped floor, but
    an emission wider than the window lifts its own reference;
  - a low-order Chebyshev continuum fit with one-sided sigma clipping,
    initialised on the lower 60% of the bins - immune to how much of the
    band the emission covers, at the price of only representing a smooth
    receiver passband;
  - a global 25th percentile - the flat-floor case.
- Grouping scale is searched too (merge gaps of 0.15, 1.0 and 4.0 times
  the narrower box): a multi-tone emission is one signal whose parts sit
  many widths apart, and only the objective can tell that from two
  separate carriers.  The hull of the strong parts is added to the
  candidate pool rather than replacing them.
- Segmentation objective: `10*log10(power_fraction^3 / bandwidth)`.  The
  whole band scores 0 dB by construction, so "everything is the signal"
  cannot win; the cube on the power share is the smallest integer power
  that prefers one box around a two-tone FSK emission over either tone.
  A fragmentation penalty of `3*log10(n_boxes)` stops a resolution that
  shatters noise into slivers from winning on one lucky sliver.
- A box narrower than `min_bandwidth_bins` analysis bins is a spectral
  LINE, not a channel: its width shrinks with the resolution instead of
  staying put.  FSK tones and OFDM pilots are lines inside one wider
  emission.
- Boxes are pruned of faint satellites inside a stronger box's shaping
  skirts, and the time extent of each comes from the waterfall energy
  inside its own band.
- An independent x^2 carrier line is computed and compared with the
  detected boxes; a carrier outside all of them adds a synthesised
  candidate and a warning rather than being discarded.
- Output is a RANKED segment list. The pipeline walks down it: committing
  to `segments[0]` is what turned a legitimate per-tone segmentation into
  a channelised bare carrier.

## S4 parameter estimation (`params/estimators.py`)
- OBW: ITU-R SM.443 99% power bandwidth on a floor-subtracted Welch PSD.
  When the floor subtraction empties the spectrum the estimator retreats
  to the bare spectrum and says which method it used, rather than
  returning `None` for a measurable quantity (defect D4).
- SNR is reported as **Es/N0** - noise counted in one symbol rate of
  bandwidth.  That is what the synchronised EVM measures, what predicts
  BER, and the only SNR that survives channelisation unchanged; the
  full-band figure is kept alongside it.  Two estimators:
  - the out-of-band noise floor (primary).  The noise power spectral
    density is read from the quiet part of the band and scaled to the
    reference bandwidth.  Completely independent of the constellation,
    which is what lets it arbitrate the moment estimator.  It needs a
    noise-only region, and says `unresolvable` when there is none.
  - M2M4 (cross-check).  It takes the signal kurtosis
    `ka = E|s|^4/(E|s|^2)^2` and solves `S^2 = (2 M2^2 - M4)/(2 - ka)`.
    Assuming `ka = 1` on a QAM signal gives an analytic ceiling -
    6.72 dB for 16QAM, 5.67 dB for 64QAM - which the estimator now
    detects against the spectral reference and reports as `saturated`
    rather than as a measurement.
- Every estimate carries a validity STATE: `valid`, `saturated` (the
  value is a floor) or `unresolvable` (the observable is absent).  A
  consumer that gates on a threshold must check the state first.
- Excess bandwidth is measured by FITTING the raised-cosine transition
  shape, not from the 99% integral (defect D3: 0.14 for a true 0.35) and
  not from threshold crossings.  Crossings fail from both ends - the 10%
  point sits in a tail the noise floor holds up, the 90% point sits in a
  passband that ripples - and both failures were measured, inflating a
  true 0.35 to 0.64 and 0.79 respectively.  The fit uses every point in
  the transition, on a PSD averaged for this purpose rather than for
  frequency resolution, and returns its residual so a poor fit can be
  reported as no measurement.  Accuracy through the whole pipeline is
  about +-0.15 on 60-frame bursts.
- CFO: strongest spectral line of x^M for M in {2,4,8}; hard limiter
  applied for constant-modulus signals only (it destroys the QAM x^4
  line).  An independent x^2 carrier line is computed alongside it and
  published for cross-checking rather than averaged in.
- Symbol rate: time-smoothed cyclic periodograms of x(t)x*(t-d) summed
  over delays d in {0,1,2,4}; candidate peaks scored by prominence over a
  median-filtered background; OBW guard rejects low-frequency envelope
  spurs; harmonics of kept candidates suppressed.
- FSK: constant envelope gate (amplitude cv <= 0.17, measured margins:
  shaped PSK/QAM >= 0.27, FSK <= 0.13); adaptive smoothing window (2..32
  samples) chosen by histogram concentration; tone count from interior
  peaks of the widened histogram; symbol rate from the spectral line of
  |d inst_f/dt| with sub-harmonic preference.
- OFDM: cyclic-prefix autocorrelation requiring both strong correlation
  and a bursty (peaky) profile, sizes >= 128/CP >= 16; final OFDM verdict
  additionally requires a noise-like envelope and *absent* single-carrier
  cumulant structure.

- S4/S6 reconciliation: after the receiver locks, the values it actually
  recovered replace the front-end estimates and every substitution is
  recorded.  The GMSK squaring lines and the OQPSK x^2 pair measure the
  symbol rate far more accurately than a blind periodogram, and the
  synchronised EVM is a real SNR measurement where the front-end figure
  may only be a bound.  The pre-reconciliation values are kept in
  `symbol_rate_norm_s4`, `carrier_offset_norm_s4` and `snr_db_s4`.

## S5 classification (`modulation/`)
- Cumulant engine: features [|C20|, |C40|, |C42|, m63] computed on
  CFO-corrected, timing-recovered symbols; references computed exactly
  from the constellation tables with the same estimators (estimator bias
  cancels); temperature-scaled softmax over weighted distances, softened
  below 15 dB SNR.
- CVNet-RF: frames of 1024 raw IQ samples, per-channel z-score, softmax
  averaged over up to 32 frames; 24 RadioML classes mapped onto the
  supported set, out-of-distribution mass reported.
  The SNR is passed to the cumulant engine only when its state is
  `valid`: the engine widens its posterior below 15 dB, and a saturated
  QAM estimate never reaches 15 dB, so the sharp mode used to be
  unreachable for exactly the constellations that need it.
- Fusion: physical evidence shapes the PRIOR (the FSK tone histogram,
  the h=0.5 squaring signature, the constant-envelope test); OFDM is the
  only class still decided without a receiver.  Then a weighted vote
  produces a ranked prior.
- Receiver-trial arbitration decides.  Candidates are trial-demodulated
  with the real S6 receiver and scored on their own family's terms:
  - linear families by EVM normalised to their own calibrated gate,
    multiplied by constellation occupancy (a sparse alphabet embeds in a
    denser one) and by a residual-bias structure score (a cloud
    quantised onto a table leaves residuals biased inside every decision
    region);
  - continuous-phase families by tone-fit ratio (M-FSK) or by how close
    the measured deviation is to what h=0.5 predicts (GMSK) - a
    MODULATION-INDEX test, not a "bigger is better" quality figure -
    gated on the constant-envelope test.
  Escalation: shortlist of three, then family representatives, then
  every remaining class, then the runner-up symbol rates.  Whenever a
  candidate wins, every SPARSER member of its family is measured too,
  because occupancy can only reject an embedding it was allowed to
  measure.
- The full ranking is published in `trial_ranking`, including candidates
  that were never trialled (`measured = false`).  "Tested and rejected"
  and "never tested" are different statements about a classification.
  The fused-prior winner and the trial winner are both named and a
  disagreement is flagged.  `--rank-all` trials everything.

## S6 demodulation (`demod/receiver.py`)

Family dispatch: psk / oqpsk / qam / apsk / ask (linear chain), fsk /
gmsk (discriminator chain), analog (audio detectors). Every path ends in
the quality gate that sets `demodulation_status` from per-modulation EVM
gates, lock flags and the rotational-concentration metric; FAILED
demodulations never feed the bit layer.

EVM gates are calibrated as `min(3GPP-derived limit, measured BER
limit)`.  The 3GPP TS 38.104 transmitter limits are a defensible
starting point but are not receiver decision limits for blind hard
slicing; a Monte-Carlo sweep through this project's own slicer measured
the EVM that actually produces 1e-3 and 1e-2 BER, and found four gates
admitting 7-16% BER.  Taking the minimum keeps whichever source was
stricter, because a too-strict gate only downgrades a good signal while
a too-loose one feeds the bit layer bits it cannot decode.
`_EVM_GATE_CALIBRATION` records both sources per modulation.

Amplitude scale is measured, not assumed.  The unit-power AGC assumes
every constellation point is used equally often and real framed traffic
never is - measured on this project's own frames the mean symbol power
is off by +2.5% for 16QAM, +5.8% for 256QAM and -5.2% for 128APSK.  A
few percent of scale is harmless for QPSK and decisive for a dense grid.
Three estimators, in order:
- the RADIUS match, applied first because it is invariant to both a
  constant rotation and a residual frequency, so it fixes the scale
  before the carrier is known and breaks the circular dependency between
  the two;
- the LATTICE PITCH, for QAM and ASK: the I/Q projections of a lattice
  constellation are periodic at the lattice pitch whatever subset of
  points the data uses, and the search span stops short of sqrt(2)
  because a square lattice rotated 45 degrees and scaled by sqrt(2) is a
  lattice again;
- a decision-directed least-squares refinement, guarded so it can never
  make the fit worse (on an unlocked constellation the nearest-point
  objective is minimised by shrinking everything onto the inner points).

Key design facts (each was verified against ground truth the hard way):
- Feedforward corrections are normalised by the taps that actually
  contributed.  A fixed 1/window scaling divides the first and last
  half-window by the full window, so any garbage at a record's edge is
  smeared back over half a window of good symbols instead of being
  averaged away: six noise-only symbols at the tail of a 1150-symbol
  32PSK record moved its EVM from 1.1% to 7.6%.  For the same reason the
  burst trim moves INWARD and the receiver discards a guard at both ends.
- Dense constellations get residual FREQUENCY tracking, not just phase:
  a fit to the block-averaged decision-directed phase error, run up a
  block-length ladder (8, 16, 32, 64, 128) so a short rung pulls in a
  large offset and a long one measures the remainder precisely.  It is
  skipped for dense PSK, whose rotationally symmetric ring makes the
  nearest-point guard blind.
- FSK, GMSK and analog have a coarse CFO path: the power centroid of the
  above-floor spectrum, which is symmetric about the carrier for all of
  them, refined by the tone centres or the squaring lines.  They report
  `cfo_applied_norm` and `cfo_confident` like every other family.
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

### Concatenated codes

An outer block code sits OUTSIDE an inner trellis code, so it is
invisible on the received stream and only appears once the inner decoder
has run.  Identifying it therefore needs a second pass over the DECODED
bits rather than another sweep of the same stream.  After the strongest
convolutional or LDPC hypotheses decode, the Reed-Solomon sweep is
re-run on their output; a `concatenated` hypothesis is only reported when
it scores higher than its own inner stage, so a spurious outer match
cannot displace a good simple answer.  An interleaver between the two
stages is handled by S8, which runs first.

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

## Known-pattern correlation (`bits/correlate.py`)
- Match counts for every bit alignment come from an FFT correlation of
  the +-1-mapped sequences, so a megabit stream against a long preamble
  stays affordable even with every ambiguity transform searched.
- The transforms searched are the ones the physical layer genuinely
  leaves behind - inversion, differential decoding, bit order - and the
  one that matched is reported, because "found under inversion" is a
  different statement from "found".
- The match threshold is raised until a chance hit is unlikely across the
  WHOLE stream: `n_alignments * P(>= k of n bits match)` must fall below
  the significance bound.  A pattern too short to clear that bar in a
  given stream is refused, with the expected number of chance matches
  quoted, rather than answered with the best of many alignments.
- Repeated hits give the frame period (median spacing, with a stability
  figure) and the payload range between consecutive occurrences.
- Every stage's bits are searched: finding the preamble after
  de-interleaving and FEC decoding but not before is evidence that those
  hypotheses were right.

## Traceability
- The report carries the SHA-256 of the SOURCE FILE as stored, not of the
  decoded samples: a digest over decoded samples would change with the
  dtype interpretation, which is itself one of the things being reported.
