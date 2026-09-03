"""RFAnalyzer: the single analysis engine behind the CLI and the GUI.

Runs S0..S11 with stage caching, analyst overrides and honest uncertainty
propagation. Overrides re-run only downstream stages (the cache keys of
untouched stages are unchanged, so their results are reused).
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

from .common.cache import StageCache, content_hash
from .common.config import Config, load_config
from .common.logging import StageTimer, get_logger, new_run_id
from .common.models import (AnalysisResult, ModulationHypothesis, Payload,
                            FECHypothesis, FrameHypothesis,
                            InterleaverHypothesis, ScramblerHypothesis)
from .conditioning import condition
from .detection import detect_signals, compute_psd
from .channelization import channelize
from .params import estimate_parameters
from .modulation import classify_modulation
from .demod import demodulate
from .bits.ambiguity import enumerate_ambiguities
from .common.models import BitStream
from .framing.frames import payload_stats
from .hypothesis import bitlayer_search
from .ingestion import load_recording

STAGE_VERSION = 3   # bump to invalidate caches when algorithms change


class RFAnalyzer:
    def __init__(self, config: Optional[Config] = None,
                 use_cache: bool = True):
        self.config = config or load_config()
        self.log = get_logger("rfanalyzer", self.config.log_level)
        self.cache = StageCache(self.config.cache_dir, enabled=use_cache)

    # ------------------------------------------------------------------
    def analyze(self, path: str, sample_rate: float = None,
                center_frequency: float = None, datatype: str = None,
                signal_index: int = 0, overrides: dict = None,
                no_ml: bool = False,
                progress: Callable[[str, float], None] = None) -> AnalysisResult:
        """Full-chain analysis of one file.

        overrides: analyst-pinned values, e.g. {"modulation": "BPSK",
        "symbol_rate_norm": 0.125, "segment": {...}}. Only stages downstream
        of an override are recomputed."""
        overrides = overrides or {}
        result = AnalysisResult(run_id=new_run_id())
        timer = StageTimer(self.log, result.stage_timings)
        prog = progress or (lambda stage, frac: None)

        # ---------------- S0 ingest -----------------------------------
        with timer.stage("S0_ingest", path=path):
            rec = load_recording(path, sample_rate, center_frequency, datatype)
            result.recording_meta = rec.meta_dict()
            result.warnings.extend(rec.warnings)
        prog("ingest", 1.0)
        data_hash = content_hash(rec.samples)

        # ---------------- S1 conditioning ------------------------------
        with timer.stage("S1_condition"):
            x, cond = condition(rec.samples)
            result.conditioning = cond
            result.warnings.extend(cond.warnings)
        prog("condition", 1.0)

        # ---------------- S2 detection ---------------------------------
        with timer.stage("S2_detect"):
            key = self.cache.key(data_hash, "detect", STAGE_VERSION,
                                 {"cfar": vars(self.config.cfar)})
            cached = self.cache.get(key)
            if cached is None:
                segments, dbg = detect_signals(x, self.config.cfar,
                                               sample_rate=rec.sample_rate)
                wf = dbg["waterfall"]
                plots = {
                    "psd": _psd_plot(x),
                    "waterfall": {
                        "db": _quantize(wf["waterfall_db"]),
                        "freq_norm": wf["freq_norm"].tolist(),
                    },
                }
                self.cache.put(key, (segments, plots))
            else:
                segments, plots = cached
            for s in segments:
                s.sample_rate = rec.sample_rate
            result.segments = segments
            result.plots.update(plots)
            if not segments:
                result.warnings.append("no signals detected above the CFAR "
                                       "threshold; analysis stopped at S2")
                return result
        prog("detect", 1.0)

        # ---------------- segment selection ----------------------------
        sel = overrides.get("signal_index", signal_index)
        sel = min(sel, len(segments) - 1)
        seg = segments[sel]
        result.selected_segment = seg.to_dict()

        # ---------------- S3 channelize --------------------------------
        with timer.stage("S3_channelize"):
            ch = channelize(x, seg)
            base = ch["samples"]
        prog("channelize", 1.0)

        # ---------------- S4 parameters --------------------------------
        with timer.stage("S4_parameters"):
            params = estimate_parameters(
                base, self.config.params,
                sample_rate=ch["sample_rate"], rate_ratio=ch["rate_ratio"])
            if "symbol_rate_norm" in overrides:
                params.symbol_rate_norm = overrides["symbol_rate_norm"]
                params.samples_per_symbol = 1.0 / params.symbol_rate_norm
                params.confidences["symbol_rate"] = {
                    "value": 1.0, "method": "analyst override",
                    "verdict": "detected"}
            result.parameters = params
        prog("parameters", 1.0)

        # ---------------- S5 modulation --------------------------------
        with timer.stage("S5_modulation"):
            if "modulation" in overrides:
                modulation = overrides["modulation"]
                result.modulation = ModulationHypothesis(
                    prediction=modulation, confidence=1.0,
                    alternatives=[[modulation, 1.0]],
                    engine_predictions={"analyst": [modulation, 1.0]},
                    constraints_applied=["analyst override"])
            else:
                mcfg = self.config.modulation
                if no_ml:
                    import copy
                    mcfg = copy.copy(mcfg)
                    mcfg.cvnet_enabled = False
                result.modulation = classify_modulation(base, params, mcfg)
                modulation = result.modulation.prediction
        prog("modulation", 1.0)

        if modulation in ("UNKNOWN", "OFDM", "OOK"):
            result.warnings.append(
                f"modulation '{modulation}' is outside the demodulation set; "
                "stopping after parameter estimation (analyst may override)")
            return result

        # ---------------- S6 demodulate --------------------------------
        with timer.stage("S6_demodulate", modulation=modulation):
            sps = params.samples_per_symbol or 8.0
            demod = demodulate(base, modulation, sps, self.config.demod,
                               cfo_norm=(params.carrier_offset_norm
                                         if not modulation.endswith("FSK") else None))
            result.demodulation = demod.to_dict()
            result.warnings.extend(demod.warnings)
            if demod.symbols is not None and len(demod.symbols):
                result.plots["constellation"] = _const_plot(demod.symbols)
            if demod.hard_bits is None or len(demod.hard_bits) < 256:
                result.warnings.append("demodulation produced too few bits "
                                       "for bit-layer analysis")
                return result
        prog("demodulate", 1.0)

        # ---------------- S7 ambiguity fan-out --------------------------
        with timer.stage("S7_ambiguity"):
            if modulation.endswith("FSK"):
                streams = _fsk_streams(demod)
            else:
                streams = enumerate_ambiguities(
                    demod.symbols, modulation,
                    self.config.bitlayer.max_ambiguity_streams)
                for s in streams:
                    if s.llrs is None:
                        s.llrs = demod.llrs
        prog("ambiguity", 1.0)

        # ---------------- S8-S10 bit-layer search -----------------------
        with timer.stage("S8_S10_bitlayer", n_streams=len(streams)):
            search = bitlayer_search(streams, self.config.bitlayer,
                                     self.config.fec, self.config.framing,
                                     logger=self.log,
                                     progress=lambda f: prog("bitlayer", f))
            result.hypotheses = search["hypotheses"]
            best = search["best"]
            if best:
                _fill_result_from_best(result, best)
        prog("bitlayer", 1.0)

        return result


# ----------------------------------------------------------------------
def _fsk_streams(demod) -> list:
    bits = demod.hard_bits
    inv = (1 - bits).astype(np.uint8)
    def diff(b):
        d = np.bitwise_xor(b[1:], b[:-1])
        return np.concatenate([[b[0]], d]).astype(np.uint8)
    return [
        BitStream(bits=bits, llrs=demod.llrs, hypothesis={"mapping": "normal"}),
        BitStream(bits=inv, hypothesis={"mapping": "inverted"}),
        BitStream(bits=diff(bits), hypothesis={"mapping": "normal", "differential": True}),
        BitStream(bits=diff(inv), hypothesis={"mapping": "inverted", "differential": True}),
    ]


def _fill_result_from_best(result: AnalysisResult, best: dict):
    cand = best["candidate"]
    wh = cand["whitener"]
    bm = best.get("scrambler_bm") or {}
    if wh["name"] != "none":
        from .scrambling.lfsr import KNOWN_WHITENERS
        winfo = KNOWN_WHITENERS.get(wh["name"], {})
        result.scrambler = ScramblerHypothesis(
            kind="known_whitening", name=wh["name"],
            polynomial=winfo.get("poly"), degree=winfo.get("degree"),
            seed=wh.get("phase"), score=1.0)
    elif bm.get("found"):
        result.scrambler = ScramblerHypothesis(
            kind="additive_lfsr", polynomial=bm["poly"], degree=bm["degree"],
            score=bm["match"])
    else:
        result.scrambler = ScramblerHypothesis(kind="none", score=0.5)

    result.interleaver = best["interleaver"]
    result.fec = best["fec"]
    if result.fec is not None:
        result.fec.decoded_bits = None   # keep result JSON-sized
    fr = best.get("frames")
    if fr:
        analysis = fr["analysis"]
        sync = analysis["sync"]
        crc = best["crc_hits"][0] if best["crc_hits"] else None
        fh = FrameHypothesis(
            frame_length_bits=fr["frame_length"],
            sync_word_hex=sync.get("hex") if sync.get("found") else None,
            sync_offset=sync.get("offset_bits"),
            column_entropy=analysis["entropy"],
            field_map=analysis["fields"],
            crc={k: (hex(v) if isinstance(v, int) and k in
                     ("poly", "init", "xorout") else v)
                 for k, v in crc.items()} if crc else None,
            score=best["hypothesis"].score)
        result.frames = fh
        # payload: bytes of the non-sync, non-crc region of each frame
        frames_mat = analysis["frames"]
        payload = _extract_payload(frames_mat, analysis, crc)
        stats = payload_stats(payload)
        result.payload = Payload(data=payload, **stats)
        if stats["likely_encrypted"]:
            result.warnings.append(
                "payload entropy is near 1 bit/bit: content is likely "
                "encrypted or compressed; no further recovery attempted")
        result.plots["entropy_map"] = [round(float(e), 3)
                                       for e in analysis["entropy"]]
    result.plots["rank_profile"] = (result.interleaver.rank_profile
                                    if result.interleaver else None)


def _extract_payload(frames_mat: np.ndarray, analysis: dict, crc) -> bytes:
    ent = analysis["entropy"]
    sync = analysis["sync"]
    start_bit = 0
    if sync.get("found"):
        start_bit = sync["offset_bits"] + sync["length_bits"]
    end_bit = frames_mat.shape[1]
    if crc:
        end_bit = min(end_bit, crc["crc_byte_offset"] * 8)
    if end_bit <= start_bit:
        start_bit, end_bit = 0, frames_mat.shape[1]
    out = bytearray()
    for row in frames_mat:
        seg = row[start_bit:end_bit]
        n8 = (len(seg) // 8) * 8
        if n8:
            out.extend(np.packbits(seg[:n8]).tobytes())
    return bytes(out)


def _psd_plot(x: np.ndarray) -> dict:
    p = compute_psd(x)
    step = max(1, len(p["freq_norm"]) // 2048)
    return {"freq_norm": p["freq_norm"][::step].tolist(),
            "psd_db": [round(float(v), 2) for v in p["psd_db"][::step]]}


def _const_plot(symbols: np.ndarray, max_points: int = 3000) -> dict:
    s = symbols[:max_points]
    return {"i": [round(float(v), 4) for v in s.real],
            "q": [round(float(v), 4) for v in s.imag]}


def _quantize(wf: np.ndarray, max_rows: int = 256, max_cols: int = 512) -> list:
    r_step = max(1, wf.shape[0] // max_rows)
    c_step = max(1, wf.shape[1] // max_cols)
    small = wf[::r_step, ::c_step]
    return [[round(float(v), 1) for v in row] for row in small]
