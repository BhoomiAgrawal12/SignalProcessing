# Architecture

## Layout

```
src/rfanalyzer/
  common/          data models, config, logging, stage cache
  ingestion/       S0: WAV/RIFF+auxi, raw IQ sniffing, SigMF, memmap loading
  conditioning/    S1: DC, IQ imbalance (Gram-Schmidt), clipping, dead air
  detection/       S2: PSD, waterfall, CFAR detection
  channelization/  S3: shift + resample + burst trimming
  params/          S4: OBW, SNR, CFO, symbol rate, FSK tones, OFDM CP
  modulation/      S5: cumulant engine, CVNet-RF adapter, fusion voter
  demod/           S6: constellations, RRC, timing, carrier, slicer, FSK
  bits/            packing (uint64), S7 ambiguity fan-out, known-pattern
                   correlation over recovered streams
  scrambling/      S7: LFSR, Berlekamp-Massey, whitener phase alignment
  gf2/             GF(2) rank engine (core primitive)
  interleaving/    S8: interleavers + blind identification
  fec/             S9: conv/Viterbi, RS, blind identification
  framing/         S10: frame period, entropy map, sync, CRC
  hypothesis/      the bit-layer search (beam + evidence + early exit)
  signatures/      SQLite signature library
  reporting/       S11: JSON/CSV/payload/SigMF/PDF/GRC exports
  pipeline.py      RFAnalyzer: S0-S11 orchestration, caching, overrides
  cli.py           command-line interface
  gui/app.py       PyQt6 analyst GUI (same engine as the CLI)
```

## Principles

1. **One engine, three frontends.** CLI, GUI and tests all call
   `RFAnalyzer.analyze()`. No business logic is duplicated.
2. **Measurement is separate from decision.** Every S4 estimate is an
   `Estimate` carrying value, confidence, method AND a validity state:
   `valid`, `saturated` (the number is a bound) or `unresolvable` (the
   observable is absent). Any consumer that gates on a threshold must
   check the state first - a saturated SNR of 6.7 dB compared against a
   15 dB threshold fires permanently, which is how the cumulant engine
   ended up running in its least confident mode for exactly the
   constellations that needed the sharpest one.
3. **The front end is a search too.** S2 emits a RANKED segment list
   rather than one box, produced by searching over analysis resolution,
   noise-floor model and grouping scale against an explicit objective
   (compact and high-contrast). S3-S6 walk down that list until one
   segment yields a modulation, recording every attempt in
   `segment_attempts`. S5 decides by trial-demodulating candidates with
   the real S6 receiver and publishes the whole ranking, including the
   candidates it never measured. The seam where the old greedy front end
   committed irreversibly - segment, then sample rate, then modulation -
   is where every measured failure lived.
4. **Search, not pipeline.** S7 emits up to 32 candidate bit streams
   (rotations x conjugation x differential). The hypothesis engine
   pre-scores them with a sparse GF(2) rank ladder, keeps a beam, and
   walks descrambler -> interleaver -> FEC -> framing -> CRC per
   candidate, accumulating evidence. A CRC pass validates a chain and
   stops the search.
5. **Linearity is the exploit.** Constellation ambiguity transforms,
   additive scrambling and interleaving are all linear/affine over GF(2),
   so rank structure survives them. That is why detection can run before
   the ambiguity is resolved: the identifiers absorb rotations as
   equivalent code descriptions (swapped generators, substream
   inversions) and resolve them at decode time via re-encode BER.
6. **Uncertainty is a first-class output.** Every estimate carries
   {value, confidence, method, state}; verdicts distinguish detected /
   estimated / inferred / classified / validated / unknown; engine
   disagreement is surfaced, never averaged away, and a classification
   says which candidates it measured and which it never tried.
7. **Caching for analyst-in-the-loop.** Stage results are cached by
   (recording content hash, stage, algorithm version, parameter hash).
   An override changes only the parameter hash of downstream stages, so
   re-running after "force BPSK" reuses ingest/detect/channelise results.

## Data flow of the bit-layer search

```
ambiguity streams (S7)
  -> cheap pre-score: sparse rank-ladder scan (structure survives
     scrambling and rotations)  ->  beam of best streams
  for each stream:
    -> rank profile -> period P
    -> whitener phase alignment at L=P (none + PN9/CCSDS/DVB/802.11)
    -> interleaver ID: factor pairs / helical / convolutional with
       block-phase offset sweep; "none" always also tried
    -> FEC ID: dual-code constancy sweep (conv), RS syndrome sweep
    -> Viterbi / RS decode (substream inversion resolved by re-encode BER)
    -> frame length -> sync realignment -> polarity -> CRC hunt
    -> evidence: pre-structure, descramble sig, interleaver sig,
       syndrome-zero rate, frame stability, CRC pass fraction
  early exit on CRC-validated chain with syndrome-zero rate > 0.95
```

## Performance notes

All GF(2) work runs on bit-packed uint64 arrays (64 bits/word, vectorised
XOR elimination); no Numba or C extension was needed to beat the targets
(see README benchmark table). Recordings larger than 64 MB are
memory-mapped; the waterfall/detection path never loads the full file.
