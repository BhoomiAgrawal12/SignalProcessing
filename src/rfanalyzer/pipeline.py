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
from .params.estimators import reconcile_with_receiver
from .modulation import classify_modulation
from .demod import demodulate
from .bits.ambiguity import enumerate_ambiguities
from .bits.correlate import best_result, correlate_streams
from .common.models import BitStream
from .framing.frames import payload_stats
from .hypothesis import bitlayer_search
from .ingestion import load_recording

STAGE_VERSION = 4   # bump to invalidate caches when algorithms change

# stage catalogue for the pipeline trace (drives the flow visualisation)
STAGES = [
    ("S0", "Ingestion & format detection"),
    ("S1", "Conditioning"),
    ("S2", "Signal detection (CFAR)"),
    ("S3", "Channelisation"),
    ("S4", "Parameter estimation"),
    ("S5", "Modulation classification"),
    ("S6", "Demodulation"),
    ("S7", "Ambiguity fan-out & descrambling"),
    ("S8", "Blind de-interleaving"),
    ("S9", "Blind FEC identification"),
    ("S10", "Framing & CRC"),
    ("S11", "Payload intelligence"),
    ("S12", "Reporting & export"),
]


class _Trace:
    """Collects one entry per pipeline stage for the flow visualisation."""

    def __init__(self, timings: dict):
        self.entries = {}
        self.timings = timings

    def mark(self, stage: str, status: str, summary: str):
        self.entries[stage] = {"status": status, "summary": summary}

    def finalize(self, stop_reason: str = None) -> list:
        out = []
        for sid, title in STAGES:
            e = self.entries.get(sid)
            if e is None:
                e = {"status": "skipped",
                     "summary": stop_reason or "not reached"}
            timing_key = {
                "S0": "S0_ingest", "S1": "S1_condition", "S2": "S2_detect",
                "S3": "S3_channelize", "S4": "S4_parameters",
                "S5": "S5_modulation", "S6": "S6_demodulate",
                "S7": "S7_ambiguity", "S8": "S8_S10_bitlayer",
                "S9": "S8_S10_bitlayer", "S10": "S8_S10_bitlayer",
                "S11": "S11_payload_intel",
            }.get(sid)
            out.append({"stage": sid, "title": title, **e,
                        "elapsed_s": round(self.timings.get(timing_key, 0), 3)
                        if timing_key and timing_key in self.timings else None})
        return out


class RFAnalyzer:
    def __init__(self, config: Optional[Config] = None,
                 use_cache: bool = True):
        self.config = config or load_config()
        self.log = get_logger("rfanalyzer", self.config.log_level)
        self.cache = StageCache(self.config.cache_dir, enabled=use_cache)

    # ------------------------------------------------------------------
    def _framing_config_with_signatures(self):
        """Framing config with the signature library's sync words filled
        in, so S10 can trim a constant run to a word the library knows
        (report §5.2).  A missing or unreadable library is not an error -
        the trimming rule falls back to dropping trailing constant
        bytes."""
        cfg = self.config.framing
        if getattr(cfg, "signature_sync_words", None):
            return cfg
        words = []
        try:
            from .signatures import SignatureDB
            for row in SignatureDB(self.config.signature_db).list_all():
                w = row.get("sync_word_hex")
                if w:
                    words.append(str(w).lower())
        except Exception as exc:            # library is optional
            self.log.debug("signature library unavailable: %s", exc)
        if not words:
            return cfg
        import copy
        cfg = copy.copy(cfg)
        cfg.signature_sync_words = sorted(set(words), key=len, reverse=True)
        return cfg

    def analyze(self, path: str, sample_rate: float = None,
                center_frequency: float = None, datatype: str = None,
                signal_index: int = 0, overrides: dict = None,
                no_ml: bool = False, rank_all: bool = False,
                preamble=None,
                progress: Callable[[str, float], None] = None) -> AnalysisResult:
        """Full-chain analysis of one file.

        overrides: analyst-pinned values, e.g. {"modulation": "BPSK",
        "symbol_rate_norm": 0.125, "segment": {...}}. Only stages downstream
        of an override are recomputed.

        S2 returns a RANKED segment list and S3-S6 are retried down that
        list until one yields a usable signal (report §9 layer A: "emits
        a ranked segment list, not segs[0]").  An explicit
        ``signal_index`` pins one segment and disables the retry.

        ``preamble`` is a known bit pattern (hex bytes or a 0/1 string)
        to correlate against every recovered bit stream; the result names
        the stream and the ambiguity transform the pattern was found
        under, the frame period implied by repeated hits, and the payload
        region between them."""
        overrides = overrides or {}
        result = AnalysisResult(run_id=new_run_id())
        timer = StageTimer(self.log, result.stage_timings)
        trace = _Trace(result.stage_timings)
        prog = progress or (lambda stage, frac: None)

        # ---------------- S0 ingest -----------------------------------
        with timer.stage("S0_ingest", path=path):
            rec = load_recording(path, sample_rate, center_frequency, datatype)
            result.recording_meta = rec.meta_dict()
            result.warnings.extend(rec.warnings)
            trace.mark("S0", "executed",
                       f"{rec.format} / {rec.datatype or 'n/a'}, "
                       f"{rec.n_samples:,} samples, sample rate "
                       f"{'{:,.0f} Hz'.format(rec.sample_rate) if rec.sample_rate else 'unknown'}")
        prog("ingest", 1.0)
        data_hash = content_hash(rec.samples)

        # ---------------- S1 conditioning ------------------------------
        with timer.stage("S1_condition"):
            x, cond = condition(rec.samples,
                                real_signal=rec.is_real_signal)
            result.conditioning = cond
            result.warnings.extend(cond.warnings)
            trace.mark("S1", "executed",
                       f"DC {'removed' if cond.dc_removed else 'negligible'}, "
                       f"clipping {cond.clipping_fraction*100:.2f}%, "
                       f"IQ {'corrected' if cond.iq_corrected else 'balanced'}")
        prog("condition", 1.0)

        # ---------------- S2 detection ---------------------------------
        with timer.stage("S2_detect"):
            key = self.cache.key(data_hash, "detect", STAGE_VERSION,
                                 {"cfar": vars(self.config.cfar)})
            cached = self.cache.get(key)
            if cached is None:
                segments, dbg = detect_signals(
                    x, self.config.cfar, sample_rate=rec.sample_rate,
                    positive_only=rec.is_real_signal)
                wf = dbg["waterfall"]
                plots = {
                    "psd": _psd_plot(x),
                    "waterfall": {
                        "db": _quantize(wf["waterfall_db"]),
                        "freq_norm": wf["freq_norm"].tolist(),
                    },
                }
                det_meta = {
                    "chosen_resolution": dbg.get("chosen_resolution"),
                    "chosen_noise_model": dbg.get("chosen_noise_model"),
                    "chosen_merge_gap_ratio": dbg.get(
                        "chosen_merge_gap_ratio"),
                    "resolutions_tried": dbg.get("resolutions_tried"),
                    "segmentation_scores": dbg.get("segmentation_scores"),
                    "carrier_cross_check": {
                        k: (round(float(v), 6) if isinstance(v, float) else v)
                        for k, v in (dbg.get("carrier_cross_check")
                                     or {}).items()},
                    "notes": dbg.get("notes", []),
                }
                self.cache.put(key, (segments, plots, det_meta))
            else:
                segments, plots, det_meta = (
                    cached if len(cached) == 3 else (*cached, {}))
            for s in segments:
                s.sample_rate = rec.sample_rate
            result.segments = segments
            result.detection = det_meta
            result.plots.update(plots)
            for note in det_meta.get("notes", []):
                result.warnings.append(f"S2: {note}")
            trace.mark("S2", "executed",
                       f"{len(segments)} signal(s) above threshold" +
                       (f"; best SNR {segments[0].snr_db:.1f} dB"
                        if segments else "") +
                       (f"; analysis resolution "
                        f"{det_meta.get('chosen_resolution')} bins, noise "
                        f"model '{det_meta.get('chosen_noise_model')}'"
                        if det_meta.get("chosen_resolution") else ""))
            if not segments:
                result.warnings.append("no signals detected above the CFAR "
                                       "threshold; analysis stopped at S2")
                result.pipeline_trace = trace.finalize(
                    "stopped at S2: no signals detected")
                return result
        prog("detect", 1.0)

        # ---------------- segment selection (ranked, with retry) --------
        # Report §9 layer A: S2 emits a RANKED segment list, and S3-S6 walk
        # down it instead of committing to segments[0].  The failure this
        # removes is not hypothetical - a multi-tone emission legitimately
        # produces one box per tone AND one box for the whole signal, and
        # the tone boxes score higher on compactness, so the emission's
        # own hull is the runner-up rather than the winner.  Committing to
        # the winner hands the channeliser a bare carrier.
        pinned = ("signal_index" in overrides) or signal_index != 0
        max_try = 1 if pinned else max(
            1, int(getattr(self.config.cfar, "max_segment_retries", 3)))
        sel = overrides.get("signal_index", signal_index)
        sel = max(0, min(sel, len(segments) - 1))
        order = [sel] + [i for i in range(len(segments)) if i != sel]
        order = order[:max_try]

        chosen = None
        for attempt, idx in enumerate(order):
            seg = segments[idx]
            with timer.stage("S3_channelize"):
                ch = channelize(x, seg)
                base = ch["samples"]
            with timer.stage("S4_parameters"):
                params = estimate_parameters(
                    base, self.config.params,
                    sample_rate=ch["sample_rate"],
                    rate_ratio=ch["rate_ratio"],
                    modulation_hint=overrides.get("modulation"),
                    analysis_band_norm=ch.get("analysis_band_norm"))
                if "symbol_rate_norm" in overrides:
                    params.symbol_rate_norm = overrides["symbol_rate_norm"]
                    params.samples_per_symbol = 1.0 / params.symbol_rate_norm
                    params.confidences["symbol_rate"] = {
                        "value": 1.0, "method": "analyst override",
                        "verdict": "detected", "state": "valid"}
            with timer.stage("S5_modulation"):
                if "modulation" in overrides:
                    modulation = overrides["modulation"]
                    modhyp = ModulationHypothesis(
                        prediction=modulation, confidence=1.0,
                        alternatives=[[modulation, 1.0]],
                        engine_predictions={"analyst": [modulation, 1.0]},
                        constraints_applied=["analyst override"],
                        prior_winner=modulation, trial_winner=modulation)
                else:
                    mcfg = self.config.modulation
                    if no_ml:
                        import copy
                        mcfg = copy.copy(mcfg)
                        mcfg.cvnet_enabled = False
                    modhyp = classify_modulation(base, params, mcfg,
                                                 rank_all=rank_all)
                    modulation = modhyp.prediction
            # S6 runs INSIDE the attempt so that a segment which
            # classifies but does not demodulate is also a reason to try
            # the next one.  Report §9 phase 3: replace the hard stops
            # with an ordered, budgeted search.  The receiver trials that
            # produced the classification have already paid most of this
            # cost, so the extra demodulation is cheap.
            demod = None
            usable = modulation not in ("UNKNOWN",)
            if usable and modulation != "OFDM":
                sps_use = params.samples_per_symbol or 8.0
                if not (1.5 <= sps_use <= 512):
                    result.warnings.append(
                        f"S4 reported {sps_use:.3f} samples/symbol, which is "
                        "outside the receiver's usable range; falling back "
                        "to 8.0 for demodulation")
                    sps_use = 8.0
                with timer.stage("S6_demodulate", modulation=modulation):
                    demod = demodulate(
                        base, modulation, sps_use, self.config.demod,
                        cfo_norm=(params.carrier_offset_norm
                                  if not modulation.endswith("FSK")
                                  else None))
                if demod.demodulation_status == "FAILED":
                    usable = False
                elif (getattr(demod, "audio", None) is None and
                      (demod.hard_bits is None or len(demod.hard_bits) < 256)):
                    usable = False
            record = {
                "attempt": attempt, "segment_id": int(seg.id),
                "f_low_norm": seg.f_low_norm, "f_high_norm": seg.f_high_norm,
                "bandwidth_norm": seg.bandwidth_norm,
                "rank_score": round(float(seg.rank_score), 3),
                "samples_after_trim": int(len(base)),
                "symbol_rate_norm": params.symbol_rate_norm,
                "snr_db": params.snr_db, "snr_state": params.snr_state,
                "modulation": modulation,
                "modulation_confidence": modhyp.confidence,
                "demodulation_status": (None if demod is None
                                        else demod.demodulation_status),
                "evm_percent": None if demod is None else demod.evm_percent,
                "accepted": bool(usable),
                "notes": list(seg.notes),
            }
            result.segment_attempts.append(record)
            chosen = (seg, idx, ch, base, params, modhyp, modulation,
                      demod)
            if usable:
                break
            if attempt + 1 < len(order):
                self.log.info(
                    "segment %d yielded no usable signal (modulation %s, "
                    "demodulation %s); trying the next ranked segment",
                    seg.id, modulation,
                    None if demod is None else demod.demodulation_status)

        seg, sel, ch, base, params, modhyp, modulation, demod = chosen
        result.selected_segment = seg.to_dict()
        result.parameters = params
        result.modulation = modhyp
        if len(result.segment_attempts) > 1:
            result.warnings.append(
                "S3-S6 were retried on "
                f"{len(result.segment_attempts)} ranked segments; segment "
                f"{seg.id} was the one carried forward "
                "(see segment_attempts)")
        trace.mark("S3", "executed",
                   f"signal {seg.id}: shifted {seg.center_norm:+.4f}, "
                   f"{len(base):,} samples after trim" +
                   (", channel filter "
                    f"+/-{ch['channel_filter_half_width_norm']:.4f}"
                    if ch.get("channel_filter_applied") else
                    ", no channel filter needed"))
        trace.mark("S4", "executed",
                   f"Rs {params.symbol_rate_norm if params.symbol_rate_norm else 'unknown'}"
                   f" (norm), Es/N0 {params.snr_db} dB"
                   f" [{params.snr_state}], OBW {params.obw99_norm}")
        prog("channelize", 1.0)
        prog("parameters", 1.0)

        prog("modulation", 1.0)

        trace.mark("S5", "executed",
                   f"{modulation} (confidence "
                   f"{result.modulation.confidence}"
                   + (f"; receiver trial ran {modhyp.trials_run} "
                      f"candidates, mode '{modhyp.trial_mode}'"
                      if modhyp.trials_run else "")
                   + (f"; PRIOR SAID {modhyp.prior_winner}"
                      if modhyp.trial_disagreement else "") + ")")
        if modhyp.trial_disagreement:
            result.warnings.append(
                f"the fused prior favoured {modhyp.prior_winner} while the "
                f"receiver trial locked {modhyp.trial_winner}; the trial "
                "wins, but the disagreement is worth an analyst's eye "
                "(see modulation.trial_ranking)")
        if modulation in ("UNKNOWN", "OFDM"):
            result.warnings.append(
                f"modulation '{modulation}' is outside the demodulation set; "
                "stopping after parameter estimation (analyst may override)")
            result.pipeline_trace = trace.finalize(
                f"stopped after S5: '{modulation}' not demodulated")
            return result

        # ---------------- S6 results ------------------------------------
        # The demodulation itself ran inside the segment loop above, so
        # that a segment which classifies but will not demodulate is a
        # reason to try the next one rather than a dead end.  What
        # follows records and interprets the result that was kept.
        if demod is not None:
            result.demodulation = demod.to_dict()
            result.warnings.extend(demod.warnings)
            if demod.symbols is not None and len(demod.symbols):
                result.plots["constellation"] = _const_plot(demod.symbols)
            # Reconcile S4 with what the receiver actually recovered.
            # The receiver measures the symbol rate and the carrier while
            # it locks - the GMSK squaring lines and the OQPSK x^2 pair
            # are far more accurate than a blind periodogram - and the
            # synchronised EVM is a real SNR measurement where the
            # front-end estimate may only be a bound.  The shipped
            # pipeline kept publishing the S4 figures, which is why
            # WAV-03 reported a 74% symbol-rate error for a signal its
            # own demodulator had timed correctly (report §11 item 7).
            notes = reconcile_with_receiver(params, demod)
            for note in notes:
                result.warnings.append(
                    f"S4/S6 reconciliation: {note['quantity']} "
                    f"{note['s4']} -> {note['s6']} ({note['reason']})")
            if getattr(demod, "eye_trace", None) is not None and \
                    len(demod.eye_trace):
                q = int(demod.lock_metrics.get("eye_sps", 8))
                tr = demod.eye_trace
                n_tr = min(100, len(tr) // (2 * q))
                result.plots["eye"] = {
                    "sps": q,
                    "traces": [[round(float(v), 4) for v in
                                tr[i * 2 * q:(i + 1) * 2 * q].real]
                               for i in range(n_tr)]}
            trace.mark("S6", "executed",
                       f"status {demod.demodulation_status}, "
                       f"{0 if demod.symbols is None else len(demod.symbols):,}"
                       f" symbols, EVM {demod.evm_percent}%, locks "
                       f"T={demod.timing_locked} C={demod.carrier_locked}")
            if getattr(demod, "audio", None) is not None:
                # analog transmission: audio out, the bit layer does not apply
                a = demod.audio
                step = max(1, len(a) // 4000)
                result.plots["audio"] = {
                    "samples": [round(float(v), 4) for v in a[::step][:4000]],
                    "decimation": step}
                result.pipeline_trace = trace.finalize(
                    "analog transmission: no bit layer")
                return result
            if demod.demodulation_status == "FAILED":
                result.warnings.append(
                    "S6 quality gate: demodulation FAILED "
                    f"(EVM {demod.evm_percent}%, locks T={demod.timing_locked} "
                    f"C={demod.carrier_locked}); the bit layer is withheld "
                    "rather than fed unreliable bits")
                result.pipeline_trace = trace.finalize(
                    "stopped by the S6 quality gate: unreliable demodulation")
                return result
            if demod.hard_bits is None or len(demod.hard_bits) < 256:
                result.warnings.append("demodulation produced too few bits "
                                       "for bit-layer analysis")
                result.pipeline_trace = trace.finalize(
                    "stopped after S6: too few bits recovered")
                return result
        else:
            result.warnings.append(
                f"no demodulator applies to '{modulation}'; the analysis "
                "stops with the parameter estimates")
            result.pipeline_trace = trace.finalize(
                f"stopped after S5: no demodulator for '{modulation}'")
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
            trace.mark("S7", "executed",
                       f"{len(streams)} candidate bit streams")
        prog("ambiguity", 1.0)

        # ---------------- S8-S10 bit-layer search -----------------------
        frames_matrix = None
        with timer.stage("S8_S10_bitlayer", n_streams=len(streams)):
            framing_cfg = self._framing_config_with_signatures()
            search = bitlayer_search(streams, self.config.bitlayer,
                                     self.config.fec, framing_cfg,
                                     logger=self.log,
                                     progress=lambda f: prog("bitlayer", f))
            result.hypotheses = search["hypotheses"]
            best = search["best"]
            if best:
                _fill_result_from_best(result, best)
                fr = best.get("frames")
                if fr:
                    frames_matrix = fr["analysis"]["frames"]
            il, fec, frh = result.interleaver, result.fec, result.frames
            scr = result.scrambler
            s7_extra = (f"; descrambler {scr.name or scr.kind}"
                        if scr else "")
            trace.mark("S7", "executed",
                       f"{len(streams)} candidate bit streams{s7_extra}")
            trace.mark("S8", "executed",
                       f"interleaver: {il.kind} {il.parameters if il.kind != 'none' else ''}"
                       if il else "no interleaver hypothesis")
            trace.mark("S9", "executed",
                       (f"FEC: {fec.family}"
                        + (f", syndrome-zero rate {fec.syndrome_zero_rate}"
                           if fec.syndrome_zero_rate is not None else ""))
                       if fec else "no FEC hypothesis")
            trace.mark("S10", "executed",
                       (f"frames {frh.frame_length_bits} bits, sync "
                        f"{frh.sync_word_hex}, CRC "
                        f"{frh.crc['name'] if frh.crc else 'none found'}")
                       if frh else "no frame structure found")
        prog("bitlayer", 1.0)

        # ---------------- S11 payload intelligence ----------------------
        if self.config.payload_intelligence.enabled and result.payload and \
                result.payload.data:
            with timer.stage("S11_payload_intel",
                             n_bytes=len(result.payload.data)):
                from .payload import analyze_payload
                crc = result.frames.crc if result.frames else None
                provenance = {
                    "crc_validated": bool(crc and
                                          crc.get("pass_fraction", 0) > 0.9),
                    "fec_syndrome_rate": (result.fec.syndrome_zero_rate
                                          if result.fec else None),
                    "demod_locked": bool(demod.carrier_locked and
                                         demod.timing_locked),
                    "demod_status": demod.demodulation_status,
                    "frame_validated": bool(result.frames is not None),
                }
                result.payload_intelligence = analyze_payload(
                    result.payload.data, frames=frames_matrix,
                    provenance=provenance,
                    config=self.config.payload_intelligence)
                pi = result.payload_intelligence
                trace.mark("S11", "executed",
                           f"{pi['summary']['classification']} "
                           f"({pi['summary']['strength']}, "
                           f"{pi['summary']['confidence']})"
                           if pi.get("available") else "no payload to analyse")
        else:
            trace.mark("S11", "skipped",
                       "disabled in config" if not
                       self.config.payload_intelligence.enabled
                       else "no payload bytes recovered")
        prog("payload_intel", 1.0)

        # ---------------- known-pattern correlation ---------------------
        # Run over EVERY stage's bits, not just the final one: finding
        # the preamble after de-interleaving and FEC decoding but not
        # before is itself evidence that those hypotheses were right, and
        # that is a stronger statement than either result alone.
        if preamble:
            streams = {}
            if demod is not None and demod.hard_bits is not None:
                streams["demodulated bits (S6)"] = demod.hard_bits
            if best:
                cand_bits = (best.get("candidate") or {}).get(
                    "whitener", {}).get("bits")
                if cand_bits is not None and len(cand_bits):
                    streams["descrambled bits (S7)"] = cand_bits
                dec = best.get("decoded_bits")
                if dec is not None and len(dec):
                    streams["FEC-decoded bits (S9)"] = dec
            if frames_matrix is not None and len(frames_matrix):
                streams["framed bits (S10)"] = np.asarray(
                    frames_matrix, dtype=np.uint8).ravel()
            if result.payload and result.payload.data:
                streams["payload bytes (S11)"] = np.unpackbits(
                    np.frombuffer(result.payload.data, dtype=np.uint8))
            try:
                found = correlate_streams(streams, preamble)
            except ValueError as exc:
                result.warnings.append(f"preamble not understood: {exc}")
                found = {}
            if found:
                winner = best_result(found)
                result.correlation = {
                    "pattern": str(preamble),
                    "streams": {k: v.to_dict() for k, v in found.items()},
                    "best_stream": winner.stream_name if winner else None,
                }
                if winner is None:
                    result.warnings.append(
                        f"the preamble '{preamble}' was not found in any "
                        "recovered bit stream at a significant match "
                        "level; see correlation.streams for the "
                        "thresholds that were applied")
                else:
                    result.warnings.append(
                        f"preamble found in {winner.stream_name} "
                        f"({winner.best_transform}): {len(winner.hits)} "
                        f"occurrence(s)"
                        + (f", period {winner.period_bits} bits"
                           if winner.period_bits else ""))

        trace.mark("S12", "ready",
                   "exports: JSON, CSV, payload, SigMF, PDF, GRC "
                   "(rendered by CLI/GUI/web)")
        result.pipeline_trace = trace.finalize()
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
    if fr and not fr.get("valid", True):
        result.warnings.append(
            "frame structure candidate rejected: insufficient evidence (" +
            ", ".join(f"{k}={v}" for k, v in
                      fr.get("validity_evidence", {}).items()) +
            "); periodicity alone does not make a frame")
        fr = None
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
    """Payload = the CRC-protected data section minus sync and header.

    Per-byte-column behaviour across frames separates the fields: the
    sync run and counter-like bytes are header; a constant byte inside
    the data section still belongs to the payload when it matches the
    payload's content class (text payloads legitimately contain constant
    label characters, which a pure entropy rule would misfile as sync
    and silently drop from every frame)."""
    sync = analysis["sync"]
    n_bits = frames_mat.shape[1]
    start_bit = 0
    if sync.get("found"):
        start_bit = sync["offset_bits"] + sync["length_bits"]
        # keep the payload byte-aligned: a constant run can end mid-byte
        # (constant MSBs of a following counter field) and a bit-shifted
        # slice would garble every downstream byte analysis
        start_bit -= start_bit % 8
    end_bit = n_bits
    if crc:
        end_bit = min(end_bit, crc["crc_byte_offset"] * 8)
    if end_bit <= start_bit:
        start_bit, end_bit = 0, n_bits
    n_bytes = (end_bit - start_bit) // 8
    if n_bytes <= 0:
        out = bytearray()
        for row in frames_mat:
            seg = row[start_bit:end_bit]
            n8 = (len(seg) // 8) * 8
            if n8:
                out.extend(np.packbits(seg[:n8]).tobytes())
        return bytes(out)
    vals = np.packbits(
        frames_mat[:, start_bit:start_bit + n_bytes * 8], axis=1)

    def _is_counter(v):
        if len(v) < 4:
            return False
        d = np.diff(v.astype(np.int16)) % 256
        top, cnt = np.unique(d, return_counts=True)
        best = int(top[np.argmax(cnt)])
        return best != 0 and cnt.max() >= 0.8 * len(d)

    def _printable_frac(v):
        return float(np.mean(((v >= 0x20) & (v < 0x7f)) |
                             (v == 9) | (v == 10) | (v == 13)))

    # core = varying, non-counter byte columns. Structured text varies
    # only in the low bits of each character, so a mean-bit-entropy
    # threshold would miss it entirely.
    core = [i for i in range(n_bytes)
            if not np.all(vals[:, i] == vals[0, i])
            and not _is_counter(vals[:, i])]
    core_text = bool(core) and float(
        np.mean([_printable_frac(vals[:, i]) for i in core])) > 0.9
    first = 0
    while first < n_bytes:
        v = vals[:, first]
        if _is_counter(v):
            first += 1
            continue
        if np.all(v == v[0]):
            if core_text and _printable_frac(v) == 1.0:
                break        # constant text character: payload label
            first += 1
            continue
        break                # varying non-counter byte: payload begins
    if first >= n_bytes:
        first = 0
    return vals[:, first:].tobytes()


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
