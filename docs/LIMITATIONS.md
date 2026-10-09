# Limitations

What Dhwani does not do, or does only partly. Each item says how the
result shows it, so a limitation is never a silent wrong answer.

## Input and front end

- **Sample rate of headerless raw IQ** cannot be measured. Unless the
  user, a SigMF sidecar or the file name states it, it stays unknown; S4
  lists PROBABLE candidates (common SDR rates at which the symbol rate
  lands on a standard baud rate), usually several and tied, never
  applied or ranked.
- **One 2^22-sample window** (about 2.1 s at 2 MS/s) is analysed: the
  one with the strongest detection among up to 16 scanned (about 67M
  samples; later samples are not scanned and the result says so), or the
  one `--start-sample` pins. Several bursts in different windows are not
  analysed together.
- **Format sniffing of headerless IQ** can misread a mostly-noise
  complex64 file as int8 (seen with a burst occupying <1% of the file);
  pass `--datatype` when the format is known.
- **Mono WAV** is converted to complex baseband at half the rate: the
  reported sample rate is fs/2 and the centre frequency is shifted by
  fs/4.
- **I/Q imbalance** is measured, not corrected by default: blind
  Gram-Schmidt assumes a circular signal and broke zero-CFO constellations.
- **DC offset** is measured on signal-free stretches. A capture with no
  quiet stretch (a continuous signal) falls back to the whole-file mean,
  which also removes the signal's own mean; the result warns.
- **Dead air** is reported, not excluded from later statistics.

## Modulation and demodulation

- **Dense PSK carrier ambiguity:** the symbol-domain carrier estimate is
  ambiguous by Rs/m (m = rotational symmetry). When the channeliser leaves
  a residual near Rs/(2m) (16PSK: Rs/32), a false lock keeps a good EVM
  with random bits; the gate then reports DEGRADED with an "ambiguous"
  warning instead of GOOD, but does not resolve it. APSK rings can alias
  at a finer step than the overall symmetry (128APSK at zero CFO: BER
  0.35, gated DEGRADED by EVM).
- **OOK** is modelled as bipolar (identical to BPSK after DC removal), so
  an OOK burst is reported as BPSK.
- **Browser engine (S0-S5 preview)** classifies from cumulants alone,
  without receiver trials; on 16QAM its |C40| reads 0.3-0.5 against 0.68
  ideal and can tip to 64QAM.
- **Unscrambled convolutional codes:** a coded but unwhitened QPSK burst
  with constant frame fields can fail timing lock (S6 FAILED, bit layer
  withheld).
- **Dense constellations near their SNR floor** can report a lookalike
  neighbour (256QAM vs 128QAM); 128APSK and 256QAM gate DEGRADED with an
  ambiguity warning. An analyst override recovers the chain.
- **OFDM** is detected and parameterised (FFT size, CP length), not
  demodulated.
- **Analog (AM/FM/SSB) is not blindly classified:** S5 has no analog
  classes, so an analog signal is demodulated to audio only when the
  analyst supplies the modulation (`--modulation FM`).
- **CVNet-RF** is advisory: its model card reports about 54% validation
  accuracy averaged over -20..+30 dB SNR on RadioML 2018.01A.

## Bit layer

- Blind interleaver and FEC identification need mostly-correct bits
  (rank-based methods); `validate_bitlayer.py` measures it at 12 and 20 dB.
- **Pseudo-random interleavers:** the period is detected and reported;
  an unknown permutation is not recovered, so the chain stops without
  frames. The 802.11a/g bit interleaver is identified by name.
- **Whitening after a permutation interleaver** (802.11, and block /
  helical over RS or LDPC): the whitener phase search needs a code period
  the interleaver hides, so these chains do not recover yet.
- **Concatenated codes:** RS outer + convolutional inner only, found when
  the inner decode has no CRC pass; no interleaver between the codes is
  searched for, and RS codewords cut by the capture edges are lost.
- **LDPC** is identified from a candidate set of known matrices; an
  arbitrary H is not reconstructed (stated in the result).
- **DVB whitening** (period 32767) is phase-searched over its first 512
  phases; when no scrambler is found and nothing validates, the result
  warns.
- Streams that never validate explore the whole beam and pay the RS/LDPC
  sweeps on every node (minutes per analysis).

## Payload and exports

- Encrypted payloads are reported as high entropy with no recovery
  attempted; that is the intended outcome.
- The GNU Radio `.grc` export targets GR 3.10 block ids, is written only
  when the sample rate is known, and is never executed here.

## Evaluation

- Every number currently produced is synthetic (`dhwani.synth`). Off-air
  scoring needs licensed recordings and `scripts/validate_offair.py`.
