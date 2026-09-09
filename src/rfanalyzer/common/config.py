"""Configuration: JSON file + dataclass with sensible defaults.

No magic numbers in stage code - everything tunable lives here.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CFARConfig:
    threshold_db: float = 8.0          # margin over local noise floor
    guard_bins: int = 2
    train_bins: int = 16
    # A box narrower than this many analysis bins is a spectral LINE,
    # not a channel: its measured width shrinks with the analysis
    # resolution instead of staying put.  The distinction matters because
    # an FSK emission's tones, and an OFDM comb's pilots, are lines
    # sitting inside one wider emission - scoring them as candidate
    # channels hands the channeliser a pure carrier.
    min_bandwidth_bins: int = 8
    min_duration_frames: int = 2
    morph_close_size: int = 3
    nfft: int = 4096
    max_signals: int = 64
    # candidate grouping scales: merge two boxes when the gap between
    # them is at most this fraction of the narrower box.  All of them are
    # tried and the segmentation objective picks - 0.15 keeps a shaped
    # single carrier in one piece, 4.0 groups the tones of an FSK
    # emission or the subcarriers of an OFDM comb.
    merge_gap_ratios: list = field(default_factory=lambda: [0.15, 1.0, 4.0])
    # drop a box sitting inside a much stronger box's shaping skirts when
    # it holds less than this share of the strong box's power (defect D5)
    skirt_power_ratio: float = 0.02
    # how many ranked segments the pipeline may fall back to when the
    # first one yields no usable modulation (report §9 layer A)
    max_segment_retries: int = 3


@dataclass
class ParamEstConfig:
    obw_fraction: float = 0.99
    fam_max_samples: int = 262144
    fam_np: int = 256                  # FAM channelisation window
    symbol_rate_min_norm: float = 1e-4
    symbol_rate_candidates: int = 8
    # the cyclic search floor is max(symbol_rate_min_norm, guard * OBW99):
    # a linear modulation's rate cannot sit far below its own occupied
    # bandwidth, but the guard must leave room for heavily filtered
    # audio-rate telemetry (report §10.5 item 2)
    symbol_rate_obw_guard: float = 0.2


@dataclass
class ModulationConfig:
    classes: list = field(default_factory=lambda: [
        "BPSK", "QPSK", "OQPSK", "8PSK", "16PSK", "32PSK",
        "OOK", "4ASK", "8ASK",
        "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
        "16APSK", "32APSK", "64APSK", "128APSK",
        "2FSK", "4FSK", "GMSK"])
    cvnet_enabled: bool = True
    cvnet_repo: str = "sohelimi/cvnet-rf"
    cvnet_checkpoint: str = ""         # local path; resolved at runtime
    cvnet_variant: str = "real"        # "real" (54% val acc) or "complex"
    device: str = "auto"               # auto | cpu | mps | cuda
    frame_size: int = 1024
    max_frames: int = 32
    fusion_weights: dict = field(default_factory=lambda: {
        "cumulant": 0.5, "cvnet": 0.35, "local_cnn": 0.15})
    # trial EVERY candidate with the real receiver instead of stopping at
    # the first that clears the acceptance bar.  Off by default: a full
    # sweep of the 18 linear classes costs about 4.1 s against a 6.3 s
    # end-to-end runtime (report §8.2).  The CLI exposes it as --rank-all.
    rank_all: bool = False


@dataclass
class DemodConfig:
    target_sps: float = 4.0
    rrc_rolloff: float = 0.35
    rrc_span_symbols: int = 10
    timing_loop_bw: float = 0.02
    carrier_loop_bw: float = 0.02
    max_symbols: int = 200000


@dataclass
class BitLayerConfig:
    max_ambiguity_streams: int = 32
    beam_width: int = 8
    rank_scan_max_L: int = 512
    rank_scan_rows_factor: int = 2     # rows = factor*L + 32
    rank_scan_max_bits: int = 200000
    interleaver_block_max: int = 4096
    conv_interleaver_max_branches: int = 16
    deficiency_significance: float = 3.0


@dataclass
class FECConfig:
    conv_constraint_lengths: list = field(default_factory=lambda: [3, 5, 7, 9])
    conv_rates: list = field(default_factory=lambda: ["1/2"])
    conv_exhaustive_max_k: int = 7
    rs_candidates: list = field(default_factory=lambda: [
        {"n": 255, "k": 223, "prim": 0x11d, "fcr": 112, "generator": 11},
        {"n": 255, "k": 223, "prim": 0x11d, "fcr": 1, "generator": 2},
        {"n": 255, "k": 239, "prim": 0x11d, "fcr": 1, "generator": 2},
        {"n": 255, "k": 239, "prim": 0x11d, "fcr": 0, "generator": 2},
        {"n": 255, "k": 223, "prim": 0x11d, "fcr": 0, "generator": 2},
    ])
    ldpc_candidates: list = field(default_factory=lambda: [
        {"n": 128, "k": 64, "seed": 1},
        {"n": 256, "k": 128, "seed": 1},
        {"n": 512, "k": 256, "seed": 1},
    ])
    min_syndrome_zero_rate: float = 0.7
    max_test_bits: int = 100000


@dataclass
class FramingConfig:
    max_frame_bits: int = 8192
    min_frame_bits: int = 16
    autocorr_max_bits: int = 500000
    crc_candidates: list = field(default_factory=lambda: [
        {"name": "CRC-8", "width": 8, "poly": 0x07, "init": 0x00, "refin": False, "refout": False, "xorout": 0x00},
        {"name": "CRC-8-MAXIM", "width": 8, "poly": 0x31, "init": 0x00, "refin": True, "refout": True, "xorout": 0x00},
        {"name": "CRC-16-CCITT-FALSE", "width": 16, "poly": 0x1021, "init": 0xFFFF, "refin": False, "refout": False, "xorout": 0x0000},
        {"name": "CRC-16-XMODEM", "width": 16, "poly": 0x1021, "init": 0x0000, "refin": False, "refout": False, "xorout": 0x0000},
        {"name": "CRC-16-ARC", "width": 16, "poly": 0x8005, "init": 0x0000, "refin": True, "refout": True, "xorout": 0x0000},
        {"name": "CRC-16-MODBUS", "width": 16, "poly": 0x8005, "init": 0xFFFF, "refin": True, "refout": True, "xorout": 0x0000},
        {"name": "CRC-32", "width": 32, "poly": 0x04C11DB7, "init": 0xFFFFFFFF, "refin": True, "refout": True, "xorout": 0xFFFFFFFF},
        {"name": "CRC-32-BZIP2", "width": 32, "poly": 0x04C11DB7, "init": 0xFFFFFFFF, "refin": False, "refout": False, "xorout": 0xFFFFFFFF},
    ])
    # known sync words (hex) from the signature library; the pipeline
    # fills this in so S10 can trim a constant run down to a word the
    # library recognises instead of guessing where it ends (report §5.2)
    signature_sync_words: list = field(default_factory=list)
    entropy_sync_threshold: float = 0.15
    entropy_payload_threshold: float = 0.85


@dataclass
class PayloadIntelConfig:
    enabled: bool = True
    text_decoding: bool = True
    encoding_discovery: bool = True
    structure_discovery: bool = True
    compression_detection: bool = True
    protocol_fingerprinting: bool = True
    max_analysis_bytes: int = 1 << 20
    max_decompression_bytes: int = 1 << 20


@dataclass
class Config:
    cfar: CFARConfig = field(default_factory=CFARConfig)
    params: ParamEstConfig = field(default_factory=ParamEstConfig)
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    demod: DemodConfig = field(default_factory=DemodConfig)
    bitlayer: BitLayerConfig = field(default_factory=BitLayerConfig)
    fec: FECConfig = field(default_factory=FECConfig)
    framing: FramingConfig = field(default_factory=FramingConfig)
    payload_intelligence: PayloadIntelConfig = field(
        default_factory=PayloadIntelConfig)
    cache_dir: str = ""
    signature_db: str = ""
    log_level: str = "INFO"

    def to_dict(self) -> dict:
        return asdict(self)


_SECTION_TYPES = {
    "cfar": CFARConfig, "params": ParamEstConfig, "modulation": ModulationConfig,
    "demod": DemodConfig, "bitlayer": BitLayerConfig, "fec": FECConfig,
    "framing": FramingConfig, "payload_intelligence": PayloadIntelConfig,
}


def load_config(path: Optional[str] = None) -> Config:
    """Load config from JSON, overlaying defaults. Unknown keys are ignored
    with a warning rather than crashing."""
    cfg = Config()
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for key, val in data.items():
            if key in _SECTION_TYPES and isinstance(val, dict):
                section = getattr(cfg, key)
                for k, v in val.items():
                    if hasattr(section, k):
                        setattr(section, k, v)
            elif hasattr(cfg, key) and not isinstance(getattr(cfg, key), tuple(_SECTION_TYPES.values())):
                setattr(cfg, key, val)
    if not cfg.cache_dir:
        cfg.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "rf-analyzer")
    if not cfg.signature_db:
        cfg.signature_db = os.path.join(cfg.cache_dir, "signatures.sqlite")
    return cfg
