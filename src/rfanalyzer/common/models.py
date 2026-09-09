"""Core data models shared by every pipeline stage.

Design notes
------------
* Every estimated quantity carries a confidence in [0, 1] and, where it
  matters, the method that produced it.  ``None`` always means *unknown* -
  never fabricate a value.
* Sample-rate-dependent quantities are stored in normalised units
  (cycles/sample) alongside absolute units, because for headerless .iq files
  the absolute sample rate is genuinely unknowable (see report §2, S0 note).
"""
from __future__ import annotations

import enum
import json
import time
import uuid
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class Verdict(str, enum.Enum):
    """How strongly the system believes a stage-level conclusion."""
    DETECTED = "detected"        # directly observed (e.g. WAV header field)
    ESTIMATED = "estimated"      # measured from the data with a known method
    INFERRED = "inferred"        # indirect conclusion from other evidence
    CLASSIFIED = "classified"    # output of a (possibly ML) classifier
    VALIDATED = "validated"      # confirmed by an objective downstream test
    UNKNOWN = "unknown"          # could not be determined; do not guess


@dataclass
class Confidence:
    """A value in [0,1] plus provenance."""
    value: float
    method: str = ""
    verdict: Verdict = Verdict.ESTIMATED

    def to_dict(self) -> dict:
        return {"value": round(float(self.value), 4), "method": self.method,
                "verdict": self.verdict.value}


class EstimateState(str, enum.Enum):
    """Whether a measurement actually measured what it claims to.

    Report §9 rule 1 - separate measurement from decisions.  An estimator
    must be able to say "I ran, but my answer is a floor" (``saturated``)
    or "the observable I need is not present" (``unresolvable``) instead
    of returning a number that reads like a measurement.
    """
    VALID = "valid"                # the number means what it says
    SATURATED = "saturated"        # the number is a bound, not a value
    UNRESOLVABLE = "unresolvable"  # no usable observable; value is None


@dataclass
class Estimate:
    """One measured quantity: value + confidence + method + state.

    ``value`` is always in the estimator's natural units and is ``None``
    when nothing could be measured.  Consumers MUST look at ``state``
    before using ``value`` as a measurement: a saturated SNR of 5.6 dB
    means "at least 5.6 dB", not "5.6 dB", and a decision rule keyed on
    ``snr < 15`` would otherwise fire permanently (report §3.3).
    """
    value: Optional[float] = None
    confidence: float = 0.0
    method: str = ""
    state: EstimateState = EstimateState.VALID
    verdict: Verdict = Verdict.ESTIMATED
    detail: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True only when the value may be used as a measured number."""
        return self.value is not None and self.state == EstimateState.VALID

    @property
    def lower_bound(self) -> Optional[float]:
        """The value read as a bound: identical for VALID, a floor for
        SATURATED, undefined for UNRESOLVABLE."""
        if self.state == EstimateState.UNRESOLVABLE:
            return None
        return self.value

    def to_dict(self) -> dict:
        d = {"value": (None if self.value is None
                       else round(float(self.value), 6)),
             "confidence": round(float(self.confidence), 4),
             "method": self.method,
             "state": self.state.value,
             "verdict": self.verdict.value}
        if self.detail:
            d["detail"] = self.detail
        return d

    def to_confidence_dict(self) -> dict:
        """Legacy ``confidences`` shape (value/method/verdict) kept so the
        GUI, the web viewer and existing exports keep working; the state
        is appended so nothing is lost."""
        return {"value": round(float(self.confidence), 4),
                "method": self.method, "verdict": self.verdict.value,
                "state": self.state.value}


def _json_default(o: Any):
    if isinstance(o, complex) or isinstance(o, np.complexfloating):
        return {"re": float(o.real), "im": float(o.imag)}
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, enum.Enum):
        return o.value
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, bytes):
        return o.hex()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def to_json(obj: Any, **kw) -> str:
    return json.dumps(obj, default=_json_default, **kw)


@dataclass
class Recording:
    """A loaded (possibly memory-mapped) IQ recording."""
    file_path: str
    format: str                        # "wav" | "raw_iq" | "sigmf"
    samples: np.ndarray                # complex64, possibly a view over a memmap
    sample_rate: Optional[float]       # Hz; None => unknown (headerless raw)
    sample_rate_source: str = "unknown"  # "wav_header" | "sigmf" | "user" | "unknown"
    center_frequency: Optional[float] = None
    center_frequency_source: str = "unknown"
    file_sha256: str = ""              # digest of the source file as stored
    file_size_bytes: int = 0
    datatype: str = ""                 # source storage dtype, e.g. "int16"
    endianness: str = ""               # "little" | "big" | ""
    channels: int = 1
    timestamp: Optional[str] = None
    format_confidence: float = 1.0
    # True for a single real-valued channel (mono WAV from an SSB/audio
    # receiver).  Its spectrum is conjugate symmetric, so S1 converts it
    # to the analytic signal and S2 searches only the positive half -
    # without that, the mirror image doubles every detection box and the
    # two halves merge into "the whole band" (report §10.2).
    is_real_signal: bool = False
    sniff_report: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(len(self.samples))

    @property
    def duration_s(self) -> Optional[float]:
        if self.sample_rate:
            return self.n_samples / self.sample_rate
        return None

    def meta_dict(self) -> dict:
        return {
            "file_path": self.file_path, "format": self.format,
            "sample_rate": self.sample_rate,
            "sample_rate_source": self.sample_rate_source,
            "center_frequency": self.center_frequency,
            "center_frequency_source": self.center_frequency_source,
            "file_sha256": self.file_sha256,
            "file_size_bytes": self.file_size_bytes,
            "datatype": self.datatype, "endianness": self.endianness,
            "channels": self.channels, "is_real_signal": self.is_real_signal,
            "n_samples": self.n_samples,
            "duration_s": self.duration_s, "timestamp": self.timestamp,
            "format_confidence": self.format_confidence,
            "sniff_report": self.sniff_report, "warnings": self.warnings,
        }


@dataclass
class ConditioningReport:
    dc_offset: complex = 0j
    dc_removed: bool = False
    iq_gain_imbalance_db: float = 0.0
    iq_phase_error_deg: float = 0.0
    image_rejection_before_db: Optional[float] = None
    image_rejection_after_db: Optional[float] = None
    iq_corrected: bool = False
    clipping_fraction: float = 0.0
    dead_air_fraction: float = 0.0
    noise_floor_db: Optional[float] = None
    analytic_conversion: bool = False   # real input -> analytic signal
    warnings: list = field(default_factory=list)


@dataclass
class SignalSegment:
    """One detected signal-of-interest in a wideband recording.

    Frequencies are stored in *normalised* units (cycles/sample, in
    [-0.5, 0.5)) so they remain meaningful when the absolute sample rate is
    unknown.  Absolute values are provided when the sample rate is known.
    """
    start_sample: int
    end_sample: int
    f_low_norm: float
    f_high_norm: float
    snr_db: float
    confidence: float = 0.0
    sample_rate: Optional[float] = None    # of the parent recording
    id: int = 0
    # --- segmentation provenance (report §9 layer A: a RANKED segment
    # list, not segs[0]) -------------------------------------------------
    method: str = ""              # which detector produced this box
    resolution_bins: int = 0      # analysis FFT length that found it
    power_fraction: float = 0.0   # share of above-floor power inside the box
    density_gain_db: float = 0.0  # 10log10(power_fraction / bandwidth): how
    #                               much denser the box is than the average
    #                               of the analysed band; 0 dB = the whole band
    rank_score: float = 0.0       # score used to order the segment list
    notes: list = field(default_factory=list)

    @property
    def center_norm(self) -> float:
        return 0.5 * (self.f_low_norm + self.f_high_norm)

    @property
    def bandwidth_norm(self) -> float:
        return self.f_high_norm - self.f_low_norm

    def absolute(self, name: str) -> Optional[float]:
        if self.sample_rate is None:
            return None
        return {"f_low": self.f_low_norm, "f_high": self.f_high_norm,
                "center": self.center_norm,
                "bandwidth": self.bandwidth_norm}[name] * self.sample_rate

    def to_dict(self) -> dict:
        d = {"id": self.id, "start_sample": self.start_sample,
             "end_sample": self.end_sample,
             "f_low_norm": self.f_low_norm, "f_high_norm": self.f_high_norm,
             "center_norm": self.center_norm,
             "bandwidth_norm": self.bandwidth_norm,
             "snr_db": round(self.snr_db, 2), "confidence": self.confidence,
             "method": self.method,
             "resolution_bins": self.resolution_bins,
             "power_fraction": round(float(self.power_fraction), 4),
             "density_gain_db": round(float(self.density_gain_db), 2),
             "rank_score": round(float(self.rank_score), 3),
             "notes": list(self.notes)}
        if self.sample_rate:
            d.update({"f_low_hz": self.absolute("f_low"),
                      "f_high_hz": self.absolute("f_high"),
                      "center_hz": self.absolute("center"),
                      "bandwidth_hz": self.absolute("bandwidth"),
                      "start_time_s": self.start_sample / self.sample_rate,
                      "end_time_s": self.end_sample / self.sample_rate})
        return d


@dataclass
class SignalParameters:
    """Physical-layer parameter estimates for one channelised signal."""
    # normalised (cycles/sample at the channelised rate) - always available
    carrier_offset_norm: Optional[float] = None
    obw99_norm: Optional[float] = None
    obw3db_norm: Optional[float] = None
    symbol_rate_norm: Optional[float] = None   # symbols/sample
    samples_per_symbol: Optional[float] = None
    # ranked alternatives from the cyclic periodogram: at low SNR data
    # lines can outrank the true rate line, so downstream hypothesis
    # tests must be able to revisit the runners-up
    symbol_rate_candidates: list = field(default_factory=list)
    # absolute - only when sample rate known
    sample_rate: Optional[float] = None
    carrier_offset_hz: Optional[float] = None
    obw99_hz: Optional[float] = None
    symbol_rate_hz: Optional[float] = None
    snr_db: Optional[float] = None
    excess_bandwidth: Optional[float] = None
    fsk_tone_count: Optional[int] = None
    fsk_deviation_norm: Optional[float] = None
    ofdm_detected: bool = False
    ofdm_fft_size: Optional[int] = None
    ofdm_cp_length: Optional[int] = None
    confidences: dict = field(default_factory=dict)  # name -> Confidence dict
    # --- measurement layer (report §9 rule 1) ---------------------------
    # name -> Estimate dict.  ``confidences`` above stays for backwards
    # compatibility; ``estimates`` is the authoritative record because it
    # carries the validity state that ``confidences`` cannot express.
    estimates: dict = field(default_factory=dict)
    # ``valid`` | ``saturated`` | ``unresolvable`` for snr_db; consumers
    # that gate on an SNR THRESHOLD must check this first (report §3.3).
    # ``snr_db`` is Es/N0 (noise in one symbol rate of bandwidth): the
    # quantity EVM measures, the one that predicts BER, and the one that
    # survives channelisation unchanged.
    snr_state: str = "valid"
    snr_method: str = ""
    snr_full_band_db: Optional[float] = None   # over the whole channel
    # what the front end measured before S6 reconciliation, kept so a
    # disagreement between S4 and S6 stays visible instead of being
    # silently overwritten (report §11 item 7)
    symbol_rate_norm_s4: Optional[float] = None
    carrier_offset_norm_s4: Optional[float] = None
    snr_db_s4: Optional[float] = None
    # one entry per quantity S6 corrected: {"quantity", "s4", "s6",
    # "method", "reason"}
    reconciliation: list = field(default_factory=list)
    # independent carrier cross-check (report §10.5 item 3)
    carrier_line_norm: Optional[float] = None
    carrier_line_quality: Optional[float] = None

    def set_estimate(self, name: str, est) -> None:
        """Record an Estimate under ``name`` in both the new ``estimates``
        map and the legacy ``confidences`` map."""
        self.estimates[name] = est.to_dict()
        self.confidences[name] = est.to_confidence_dict()

    def estimate_state(self, name: str) -> str:
        return (self.estimates.get(name) or {}).get("state", "valid")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


@dataclass
class TrialRank:
    """One row of the receiver-trial ranking table (report §8.3).

    Every candidate the classifier considered appears here, including the
    ones that were never trialled - those carry ``measured=False`` so an
    analyst can tell "tested and rejected" from "never tested", which is
    exactly the distinction the IQ-02 misclassification hid.
    """
    candidate: str
    prior: float = 0.0            # fused-prior probability before trials
    measured: bool = False        # was the real S6 receiver actually run?
    score: Optional[float] = None  # trial score; None = did not lock
    evm_percent: Optional[float] = None
    gate_good: Optional[float] = None
    gate_ratio: Optional[float] = None   # evm / gate_good; < 1 is a clean lock
    occupancy: Optional[float] = None
    structure: Optional[float] = None
    status: str = "not measured"   # GOOD | DEGRADED | FAILED | not measured
    samples_per_symbol: Optional[float] = None
    family: str = ""
    metrics: dict = field(default_factory=dict)   # family-specific extras
    reason: str = ""               # why it did not lock / was not measured

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ModulationHypothesis:
    prediction: str
    confidence: float
    alternatives: list = field(default_factory=list)   # [(label, prob), ...]
    engine_predictions: dict = field(default_factory=dict)  # engine -> (label, conf)
    classifier_agreement: bool = True
    constraints_applied: list = field(default_factory=list)
    in_distribution: bool = True
    # --- receiver-trial evidence (report §8) ----------------------------
    trial_ranking: list = field(default_factory=list)   # list[TrialRank]
    prior_winner: Optional[str] = None      # top of the fused prior
    trial_winner: Optional[str] = None      # top of the receiver trial
    trial_disagreement: bool = False        # prior and trial disagree
    trials_run: int = 0
    trial_mode: str = "shortlist"           # shortlist | escalated | all

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["trial_ranking"] = [r.to_dict() if hasattr(r, "to_dict") else r
                              for r in self.trial_ranking]
        return d


@dataclass
class DemodulationResult:
    modulation: str
    symbols: np.ndarray = field(default=None, repr=False)  # complex, post-sync
    hard_bits: np.ndarray = field(default=None, repr=False)  # uint8 0/1
    llrs: np.ndarray = field(default=None, repr=False)       # float, >0 => bit 0
    evm_percent: Optional[float] = None
    carrier_locked: bool = False
    timing_locked: bool = False
    cfo_applied_norm: float = 0.0
    samples_per_symbol: float = 0.0
    lock_metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    eye_trace: np.ndarray = field(default=None, repr=False)
    audio: np.ndarray = field(default=None, repr=False)
    demodulation_status: str = "FAILED"      # GOOD | DEGRADED | FAILED
    # --- values the receiver recovered itself, for S4 reconciliation ----
    # (report §11 item 7: WAV-03's GMSK demodulator found the right symbol
    # rate internally while S4's published figure stayed 74% wrong)
    cfo_confident: bool = False        # cfo_applied_norm is a measurement
    symbol_rate_norm_recovered: Optional[float] = None
    symbol_rate_confident: bool = False

    def to_dict(self) -> dict:
        return {"modulation": self.modulation,
                "demodulation_status": self.demodulation_status,
                "n_symbols": 0 if self.symbols is None else int(len(self.symbols)),
                "n_bits": 0 if self.hard_bits is None else int(len(self.hard_bits)),
                "evm_percent": self.evm_percent,
                "carrier_locked": self.carrier_locked,
                "timing_locked": self.timing_locked,
                "cfo_applied_norm": self.cfo_applied_norm,
                "cfo_confident": self.cfo_confident,
                "symbol_rate_norm_recovered": self.symbol_rate_norm_recovered,
                "symbol_rate_confident": self.symbol_rate_confident,
                "samples_per_symbol": self.samples_per_symbol,
                "lock_metrics": self.lock_metrics, "warnings": self.warnings}


@dataclass
class BitStream:
    """A candidate bit stream plus the transform hypothesis that produced it."""
    bits: np.ndarray                      # uint8 array of 0/1
    hypothesis: dict = field(default_factory=dict)
    llrs: Optional[np.ndarray] = None
    score: float = 0.0

    def __len__(self):
        return len(self.bits)


@dataclass
class ScramblerHypothesis:
    kind: str                              # "additive_lfsr" | "known_whitening" | "none"
    polynomial: Optional[int] = None       # taps as integer bitmask (x^0 = LSB)
    degree: Optional[int] = None
    seed: Optional[int] = None
    name: str = ""
    score: float = 0.0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.polynomial is not None:
            d["polynomial_hex"] = hex(self.polynomial)
        return d


@dataclass
class InterleaverHypothesis:
    kind: str            # "none" | "block" | "convolutional" | "helical" | "pseudo_random"
    parameters: dict = field(default_factory=dict)
    period: Optional[int] = None
    score: float = 0.0
    p_value: Optional[float] = None
    rank_profile: Optional[dict] = None   # {"L": [...], "deficiency": [...]}

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.rank_profile and len(self.rank_profile.get("L", [])) > 512:
            d["rank_profile"] = {k: list(v)[:512] for k, v in self.rank_profile.items()}
        return d


@dataclass
class FECHypothesis:
    family: str          # "none" | "convolutional" | "reed_solomon" | "ldpc" | "unknown"
    parameters: dict = field(default_factory=dict)
    code_rate: Optional[float] = None
    syndrome_zero_rate: Optional[float] = None
    decoded_bits: Optional[np.ndarray] = field(default=None, repr=False)
    pre_fec_ber: Optional[float] = None
    score: float = 0.0

    def to_dict(self) -> dict:
        return {"family": self.family, "parameters": self.parameters,
                "code_rate": self.code_rate,
                "syndrome_zero_rate": self.syndrome_zero_rate,
                "n_decoded_bits": 0 if self.decoded_bits is None else int(len(self.decoded_bits)),
                "pre_fec_ber": self.pre_fec_ber, "score": self.score}


@dataclass
class FrameHypothesis:
    frame_length_bits: Optional[int] = None
    sync_word_hex: Optional[str] = None
    sync_offset: Optional[int] = None
    header_bits: Optional[int] = None
    payload_bits: Optional[int] = None
    crc: Optional[dict] = None            # {"name","width","poly","init","refin","refout","xorout","passes","total"}
    column_entropy: Optional[np.ndarray] = field(default=None, repr=False)
    field_map: list = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict:
        d = {"frame_length_bits": self.frame_length_bits,
             "sync_word_hex": self.sync_word_hex, "sync_offset": self.sync_offset,
             "header_bits": self.header_bits, "payload_bits": self.payload_bits,
             "crc": self.crc, "field_map": self.field_map, "score": self.score}
        if self.column_entropy is not None:
            d["column_entropy"] = [round(float(x), 4) for x in self.column_entropy]
        return d


@dataclass
class Payload:
    data: bytes
    entropy_bits_per_bit: Optional[float] = None
    printable_fraction: Optional[float] = None
    likely_encrypted: bool = False

    def to_dict(self) -> dict:
        return {"n_bytes": len(self.data), "hex": self.data[:4096].hex(),
                "entropy_bits_per_bit": self.entropy_bits_per_bit,
                "printable_fraction": self.printable_fraction,
                "likely_encrypted": self.likely_encrypted}


@dataclass
class AnalysisHypothesis:
    """One node in the hypothesis search tree."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    stage: str = ""
    assumptions: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)    # name -> float
    score: float = 0.0
    status: str = "open"                            # open | validated | rejected
    outputs: dict = field(default_factory=dict)

    def add_evidence(self, name: str, value: float, weight: float = 1.0):
        self.evidence[name] = float(value)
        self.score = float(sum(self.evidence.values()) / max(1, len(self.evidence)))

    def to_dict(self) -> dict:
        return {"id": self.id, "parent_id": self.parent_id, "stage": self.stage,
                "assumptions": self.assumptions, "evidence": self.evidence,
                "score": round(self.score, 4), "status": self.status}


@dataclass
class AnalysisResult:
    """Everything the pipeline produced for one signal of interest."""
    run_id: str = ""
    created: float = field(default_factory=time.time)
    recording_meta: dict = field(default_factory=dict)
    conditioning: Optional[ConditioningReport] = None
    segments: list = field(default_factory=list)
    selected_segment: Optional[dict] = None
    # how S2 arrived at that list: which analysis resolution and noise
    # model won, what the carrier cross-check said, what it warned about
    detection: dict = field(default_factory=dict)
    # one entry per segment S3-S6 was tried on, in the order tried
    segment_attempts: list = field(default_factory=list)
    parameters: Optional[SignalParameters] = None
    modulation: Optional[ModulationHypothesis] = None
    demodulation: Optional[dict] = None
    scrambler: Optional[ScramblerHypothesis] = None
    interleaver: Optional[InterleaverHypothesis] = None
    fec: Optional[FECHypothesis] = None
    frames: Optional[FrameHypothesis] = None
    payload: Optional[Payload] = None
    payload_intelligence: Optional[dict] = None
    # known-pattern correlation, when the analyst supplied a preamble:
    # {"pattern": ..., "streams": {name: result}, "best": name}
    correlation: Optional[dict] = None
    hypotheses: list = field(default_factory=list)
    stage_timings: dict = field(default_factory=dict)
    pipeline_trace: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    plots: dict = field(default_factory=dict)      # name -> plot data for GUI/web

    def to_dict(self) -> dict:
        def opt(x):
            if x is None:
                return None
            return x.to_dict() if hasattr(x, "to_dict") else dataclasses.asdict(x)
        return {
            "run_id": self.run_id, "created": self.created,
            "recording": self.recording_meta,
            "conditioning": None if self.conditioning is None else dataclasses.asdict(self.conditioning),
            "segments": [s.to_dict() if hasattr(s, "to_dict") else s for s in self.segments],
            "selected_segment": self.selected_segment,
            "detection": self.detection,
            "segment_attempts": self.segment_attempts,
            "parameters": opt(self.parameters),
            "modulation": opt(self.modulation),
            "demodulation": self.demodulation,
            "scrambler": opt(self.scrambler),
            "interleaver": opt(self.interleaver),
            "fec": opt(self.fec),
            "frames": opt(self.frames),
            "payload": opt(self.payload),
            "payload_intelligence": self.payload_intelligence,
            "correlation": self.correlation,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "stage_timings": {k: round(v, 3) for k, v in self.stage_timings.items()},
            "pipeline_trace": self.pipeline_trace,
            "warnings": self.warnings,
            "plots": self.plots,
        }

    def to_json(self, **kw) -> str:
        return to_json(self.to_dict(), **kw)
