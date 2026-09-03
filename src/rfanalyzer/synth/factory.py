"""Synthetic waveform factory (report §8, figure 3): bits -> frame ->
FEC -> interleave -> scramble -> modulate -> pulse-shape -> impair -> IQ,
with the full ground-truth label set preserved at every step.

This is the regression-test backbone: every pipeline stage is scored
against these labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy import signal as sig

from ..demod.constellations import CONSTELLATIONS, bits_to_iq_symbols
from ..demod.filters import rrc_taps
from ..fec.conv import ConvCode, conv_encode
from ..fec.rs import RSCode, symbols_to_bits
from ..framing.crc import crc_compute
from ..interleaving.interleavers import (block_interleave, conv_interleave,
                                         helical_interleave, pn_interleave)
from ..scrambling.lfsr import additive_scramble, KNOWN_WHITENERS
from . import impairments as imp


@dataclass
class GroundTruth:
    modulation: str = "QPSK"
    samples_per_symbol: float = 8.0
    symbol_rate_norm: float = 0.125
    rolloff: float = 0.35
    snr_db: float = 30.0
    cfo_norm: float = 0.0
    phase_offset: float = 0.0
    fec: dict = field(default_factory=lambda: {"family": "none"})
    interleaver: dict = field(default_factory=lambda: {"kind": "none"})
    scrambler: dict = field(default_factory=lambda: {"kind": "none"})
    frame: dict = field(default_factory=dict)
    payloads: list = field(default_factory=list)
    info_bits: Optional[list] = None
    coded_bits_len: int = 0
    fsk_deviation_norm: float = 0.0
    impairments: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class WaveformFactory:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    # ---------------- frame construction ---------------------------------
    def build_frames(self, n_frames: int = 60, payload_len: int = 6,
                     sync: int = 0xEB90) -> tuple:
        frames, payloads = [], []
        for i in range(n_frames):
            payload = bytes(self.rng.integers(32, 127, payload_len,
                                              dtype=np.uint8))
            payloads.append(payload.hex())
            body = sync.to_bytes(2, "big") + bytes([i & 0xFF, 0x01]) + payload
            crc = crc_compute(body, 16, 0x1021, 0xFFFF, False, False, 0)
            frame = body + crc.to_bytes(2, "big")
            frames.append(np.unpackbits(np.frombuffer(frame, dtype=np.uint8)))
        bits = np.concatenate(frames).astype(np.uint8)
        frame_meta = {"sync_hex": f"{sync:04x}", "frame_bits": len(frames[0]),
                      "crc": "CRC-16-CCITT-FALSE", "payload_len": payload_len}
        return bits, frame_meta, payloads

    # ---------------- full stack ------------------------------------------
    def generate(self, modulation: str = "QPSK", sps: float = 8.0,
                 rolloff: float = 0.35, snr_db: float = 30.0,
                 cfo_norm: float = 0.0, phase_offset: float = 0.0,
                 fec: dict = None, interleaver: dict = None,
                 scrambler: dict = None, n_frames: int = 60,
                 impair: dict = None, n_pad_noise: int = 2000) -> tuple:
        """Returns (iq complex64, GroundTruth)."""
        gt = GroundTruth(modulation=modulation, samples_per_symbol=sps,
                         symbol_rate_norm=1.0 / sps, rolloff=rolloff,
                         snr_db=snr_db, cfo_norm=cfo_norm,
                         phase_offset=phase_offset)
        info_bits, frame_meta, payloads = self.build_frames(n_frames)
        gt.frame = frame_meta
        gt.payloads = payloads
        gt.info_bits = info_bits.tolist()

        bits = info_bits
        # FEC
        fec = fec or {"family": "none"}
        gt.fec = dict(fec)
        if fec["family"] == "convolutional":
            code = ConvCode(fec.get("K", 7),
                            tuple(fec.get("generators", (0o171, 0o133))))
            bits = conv_encode(bits, code, terminate=False)
            gt.fec.update({"K": code.K,
                           "generators_octal": [oct(g) for g in code.generators],
                           "rate": "1/2"})
        elif fec["family"] == "reed_solomon":
            rs = RSCode(fec.get("n", 255), fec.get("k", 223),
                        fcr=fec.get("fcr", 1), generator=fec.get("generator", 2))
            data = np.packbits(bits)
            k = rs.k
            pad = (-len(data)) % k
            data = np.concatenate([data, np.zeros(pad, dtype=np.uint8)])
            cws = [rs.encode(bytes(data[i:i + k])) for i in range(0, len(data), k)]
            bits = np.unpackbits(np.frombuffer(b"".join(cws), dtype=np.uint8))
            gt.fec.update({"n": rs.n, "k": rs.k, "fcr": rs.fcr})
        gt.coded_bits_len = len(bits)

        # interleave
        interleaver = interleaver or {"kind": "none"}
        gt.interleaver = dict(interleaver)
        if interleaver["kind"] == "block":
            bits = block_interleave(bits, interleaver["rows"], interleaver["cols"])
        elif interleaver["kind"] == "convolutional":
            bits = conv_interleave(bits, interleaver["branches"], interleaver["delay"])
        elif interleaver["kind"] == "helical":
            bits = helical_interleave(bits, interleaver["rows"],
                                      interleaver["cols"], interleaver["step"])
        elif interleaver["kind"] == "pseudo_random":
            perm = self.rng.permutation(interleaver.get("period", 128))
            bits = pn_interleave(bits, perm)
            gt.interleaver["permutation"] = perm.tolist()

        # scramble
        scrambler = scrambler or {"kind": "none"}
        gt.scrambler = dict(scrambler)
        if scrambler["kind"] == "known_whitening":
            w = KNOWN_WHITENERS[scrambler["name"]]
            bits = additive_scramble(bits, w["poly"], w["seed"])
            gt.scrambler.update(w)

        # modulate
        if modulation in CONSTELLATIONS:
            symbols = bits_to_iq_symbols(bits, modulation)
            iq = self._pulse_shape(symbols, sps, rolloff)
        elif modulation.endswith("FSK"):
            order = int(modulation[0])
            iq, dev = self._fsk(bits, order, sps)
            gt.fsk_deviation_norm = dev
        else:
            raise ValueError(f"unsupported modulation {modulation}")

        # impairments
        impair = impair or {}
        gt.impairments = dict(impair)
        if cfo_norm or phase_offset:
            iq = imp.freq_offset(iq, cfo_norm, phase_offset)
        if impair.get("phase_noise_std"):
            iq = imp.phase_noise(iq, impair["phase_noise_std"], self.rng)
        if impair.get("timing_frac"):
            iq = imp.timing_offset(iq, impair["timing_frac"])
        if impair.get("clock_ppm"):
            iq = imp.clock_offset(iq, impair["clock_ppm"])
        if impair.get("iq_gain_db") or impair.get("iq_phase_deg"):
            iq = imp.iq_imbalance(iq, impair.get("iq_gain_db", 0),
                                  impair.get("iq_phase_deg", 0))
        if impair.get("multipath_taps"):
            iq = imp.multipath(iq, np.array(impair["multipath_taps"]))
        iq = imp.awgn(iq, snr_db, self.rng)
        if impair.get("dc_offset"):
            iq = imp.dc_offset(iq, impair["dc_offset"])
        # noise padding before/after so detection has work to do
        if n_pad_noise:
            p = (np.abs(iq) ** 2).mean() / (10 ** (snr_db / 10))
            pad = (self.rng.normal(0, np.sqrt(p / 2), (n_pad_noise, 2))
                   @ np.array([1, 1j]))
            iq = np.concatenate([pad, iq, pad])
        return iq.astype(np.complex64), gt

    def _pulse_shape(self, symbols: np.ndarray, sps: float, beta: float) -> np.ndarray:
        n_int = int(round(sps))
        up = np.zeros(len(symbols) * n_int, dtype=np.complex128)
        up[::n_int] = symbols
        taps = rrc_taps(n_int, 10, beta)
        x = sig.fftconvolve(up, taps, mode="full")[len(taps)//2 : len(taps)//2 + len(up)]
        if abs(sps - n_int) > 1e-9:
            # non-integer sps: fractional resample
            from fractions import Fraction
            fr = Fraction(sps / n_int).limit_denominator(200)
            x = sig.resample_poly(x, fr.numerator, fr.denominator)
        return x

    def _fsk(self, bits: np.ndarray, order: int, sps: float) -> tuple:
        k = int(np.log2(order))
        n_sym = len(bits) // k
        b = bits[: n_sym * k].reshape(n_sym, k)
        weights = 1 << np.arange(k - 1, -1, -1)
        syms = (b * weights).sum(axis=1)
        # tone spacing = symbol_rate (modulation index 1.0), centred
        rs = 1.0 / sps
        dev = rs
        tones = (syms - (order - 1) / 2) * dev
        inst_f = np.repeat(tones, int(round(sps)))
        phase = 2 * np.pi * np.cumsum(inst_f)
        return np.exp(1j * phase), dev


def write_sigmf(iq: np.ndarray, gt: GroundTruth, base_path: str,
                sample_rate: float = None):
    """Write .sigmf-data (cf32_le) + .sigmf-meta with ground truth in a
    custom namespace so automated scoring can read it back."""
    iq.astype(np.complex64).tofile(base_path + ".sigmf-data")
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:version": "1.0.0",
            "core:description": "rf-analyzer synthetic waveform",
            **({"core:sample_rate": sample_rate} if sample_rate else {}),
            "rfanalyzer:ground_truth": json.loads(gt.to_json()),
        },
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }
    with open(base_path + ".sigmf-meta", "w") as f:
        json.dump(meta, f, indent=2)
