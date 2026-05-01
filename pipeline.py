"""
pipeline.py — Shared execution core for the AD/ADRD multi-agent pipeline.

Both main_c.py (batch) and api.py (REST) delegate to run_pipeline().
This module owns: parallel agent dispatch, per-agent timeouts, error collection,
synthesizer invocation, LLM call accounting, and PipelineResult construction.
"""

import concurrent.futures
import json
import time
from datetime import datetime, timezone

from agents.clintext_agent import run_clintext_agent
from agents.diagnosis_agent import run_diagnosis_agent
from agents.medication_agent import run_medication_agent
from agents.synthesizer_agent import run_synthesizer_agent
from logger import get_logger
from schemas import (
    ClinTextAgentOutput,
    DiagnosisAgentOutput,
    MedicationAgentOutput,
    PipelineResult,
)

logger = get_logger(__name__)

_AGENT_TIMEOUT_S   = 30     # wall-clock seconds each future is allowed before fallback
_LLM_COST_PER_CALL = 0.002  # USD proxy per LLM call

EVIDENCE_COLUMNS = [
    "note_id", "run_id",
    "clin_evidence_list", "clin_reasoning", "clin_assessment",
    "med_medications", "med_status", "med_source", "med_reasoning", "med_assessment",
    "dx_matched_codes", "dx_is_current_diagnosis", "dx_reasoning", "dx_assessment",
]

# Substrings in agent reasoning that flag a deterministic fallback was used
_AGENT_ERROR_MARKERS = (
    "Agent error",
    "Agent timeout",
    "LLM scan failed",
    "LLM context check failed",
    "LLM check failed",
)


# ─── Per-agent timeout fallbacks ─────────────────────────────────────────────

def _dx_timeout_fallback() -> DiagnosisAgentOutput:
    return DiagnosisAgentOutput(
        dx_found=False,
        matched_codes=[],
        is_current_diagnosis=False,
        confidence="low",
        confidence_score=0.0,
        reasoning="Agent timeout: DiagnosisAgent exceeded 30s limit.",
        assessment="Agent timeout — defaulting negative.",
    )


def _med_timeout_fallback() -> MedicationAgentOutput:
    return MedicationAgentOutput(
        meds_found=False,
        medications=[],
        status="none",
        source="none",
        confidence="low",
        confidence_score=0.0,
        reasoning="Agent timeout: MedicationAgent exceeded 30s limit.",
        assessment="Agent timeout — defaulting negative.",
    )


def _clin_timeout_fallback() -> ClinTextAgentOutput:
    return ClinTextAgentOutput(
        symptoms_found=False,
        evidence_list=[],
        confidence="low",
        confidence_score=0.0,
        reasoning="Agent timeout: ClinTextAgent exceeded 30s limit.",
        assessment="Agent timeout — defaulting negative.",
    )


# ─── Core pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    text: str,
    icd_codes: str,
    llm,
    kb=None,
    run_id: str = "",
    note_id: str = "",
    subject_id: str = "",
    ground_truth: str = "",
    ground_truth_subtype: str = "",
    include_detail: bool = False,
) -> tuple[PipelineResult, dict | None, dict]:
    """
    Execute the full 3-agent + synthesizer pipeline for one clinical note.

    Args:
        text:                  Full clinical note text.
        icd_codes:             Pipe-separated ICD-9/10 codes (e.g. 'F0280|G309').
        llm:                   LangChain LLM instance.
        kb:                    Optional KnowledgeBase for RAG (ClinTextAgent only).
        run_id:                Batch run identifier; empty string for API calls.
        note_id / subject_id:  EHR identifiers; empty strings for API calls.
        ground_truth /
        ground_truth_subtype:  Gold labels; empty strings for API calls.
        include_detail:        When True, second element of the tuple contains raw
                               per-agent outputs (evidence_list, medications, etc.).
                               When False (default), second element is None.

    Returns:
        (PipelineResult, dict | None) — PipelineResult always populated;
        detail dict only when include_detail=True.
    """
    t_start = time.perf_counter()
    row     = {"text": text, "all_icd_codes": icd_codes}
    agent_errors: list[str] = []

    # ── Parallel execution: Agents 1–3 ──────────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_dx   = executor.submit(run_diagnosis_agent,  row, llm)
        future_med  = executor.submit(run_medication_agent, row, llm)
        future_clin = executor.submit(run_clintext_agent,   row, llm, kb)

        try:
            dx_result = future_dx.result(timeout=_AGENT_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.warning("DiagnosisAgent timed out after %ds", _AGENT_TIMEOUT_S)
            agent_errors.append("DiagnosisAgent")
            dx_result = _dx_timeout_fallback()

        try:
            med_result = future_med.result(timeout=_AGENT_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.warning("MedicationAgent timed out after %ds", _AGENT_TIMEOUT_S)
            agent_errors.append("MedicationAgent")
            med_result = _med_timeout_fallback()

        try:
            clin_result = future_clin.result(timeout=_AGENT_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.warning("ClinTextAgent timed out after %ds", _AGENT_TIMEOUT_S)
            agent_errors.append("ClinTextAgent")
            clin_result = _clin_timeout_fallback()

    logger.info(
        "  [L3-DiagnosisAgent]  dx_found=%s current=%s conf=%s codes=%s",
        dx_result.dx_found, dx_result.is_current_diagnosis,
        dx_result.confidence, dx_result.matched_codes,
    )
    logger.info(
        "  [L2-MedicationAgent] meds_found=%s status=%s meds=%s source=%s",
        med_result.meds_found, med_result.status,
        med_result.medications, med_result.source,
    )
    logger.info(
        "  [L1-ClinTextAgent]   symptoms_found=%s conf=%s evidence_count=%d",
        clin_result.symptoms_found, clin_result.confidence,
        len(clin_result.evidence_list),
    )

    # ── Error-marker check (supplements timeout errors already collected) ─────
    if "ClinTextAgent" not in agent_errors and any(
        m in clin_result.reasoning for m in _AGENT_ERROR_MARKERS
    ):
        agent_errors.append("ClinTextAgent")
    if "MedicationAgent" not in agent_errors and any(
        m in med_result.reasoning for m in _AGENT_ERROR_MARKERS
    ):
        agent_errors.append("MedicationAgent")
    if "DiagnosisAgent" not in agent_errors and any(
        m in dx_result.reasoning for m in _AGENT_ERROR_MARKERS
    ):
        agent_errors.append("DiagnosisAgent")

    # ── Serial: Synthesizer ───────────────────────────────────────────────────
    synth = run_synthesizer_agent(dx_result, med_result, clin_result, llm, text=text)

    logger.info(
        "  [Synthesizer]  prediction=%s subtype=%s confidence=%s mode=%s",
        synth.final_prediction, synth.subtype, synth.confidence, synth.synthesis_mode,
    )
    logger.info("  [Causal Chain] %s...", synth.causal_chain[:200])
    if synth.discrepancy:
        logger.warning("  [Discrepancy]  %s", synth.discrepancy)
    if synth.overruled_agents:
        logger.warning("  [Overruled]    %s", synth.overruled_agents)

    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── LLM call accounting ───────────────────────────────────────────────────
    calls_clin  = 1  # ClinTextAgent always calls LLM when text is non-empty
    calls_med   = 0 if med_result.source == "structured" else 1
    calls_dx    = 0 if not dx_result.matched_codes else 1
    calls_synth = 0 if synth.synthesis_mode in (
        "consensus_positive", "consensus_negative"
    ) else 1
    llm_calls = calls_clin + calls_med + calls_dx + calls_synth

    degraded_mode = bool(agent_errors) or synth.causal_chain.startswith("[Fallback")

    pipeline_result = PipelineResult(
        note_id               = note_id,
        subject_id            = subject_id,
        ground_truth          = ground_truth,
        ground_truth_subtype  = ground_truth_subtype,
        dx_found              = dx_result.dx_found,
        meds_found            = med_result.meds_found,
        symptoms_found        = clin_result.symptoms_found,
        final_prediction      = synth.final_prediction,
        subtype               = synth.subtype,
        confidence            = synth.confidence,
        synthesis_mode        = synth.synthesis_mode,
        contributing_agents   = list(synth.contributing_agents),
        causal_chain          = synth.causal_chain,
        discrepancy           = synth.discrepancy,
        overruled_agents      = list(synth.overruled_agents),
        reason                = synth.reason,
        confidence_score_clin = synth.confidence_score_clin,
        confidence_score_med  = synth.confidence_score_med,
        confidence_score_dx   = synth.confidence_score_dx,
        confidence_correction = synth.confidence_correction,
        latency_ms            = round(latency_ms, 2),
        llm_calls             = llm_calls,
        estimated_cost_usd_proxy = round(llm_calls * _LLM_COST_PER_CALL, 4),
        degraded_mode         = degraded_mode,
        agent_errors          = agent_errors,
        run_id                = run_id,
        timestamp             = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        text_snippet          = text[:200],
    )

    evidence_record = {
        "note_id":                 note_id,
        "run_id":                  run_id,
        "clin_evidence_list":      json.dumps(list(clin_result.evidence_list)),
        "clin_reasoning":          clin_result.reasoning,
        "clin_assessment":         clin_result.assessment,
        "med_medications":         json.dumps(list(med_result.medications)),
        "med_status":              med_result.status,
        "med_source":              med_result.source,
        "med_reasoning":           med_result.reasoning,
        "med_assessment":          med_result.assessment,
        "dx_matched_codes":        json.dumps(list(dx_result.matched_codes)),
        "dx_is_current_diagnosis": dx_result.is_current_diagnosis,
        "dx_reasoning":            dx_result.reasoning,
        "dx_assessment":           dx_result.assessment,
    }

    if not include_detail:
        return pipeline_result, None, evidence_record

    agent_detail = {
        "clin": {
            "evidence_list":    list(clin_result.evidence_list),
            "reasoning":        clin_result.reasoning,
            "assessment":       clin_result.assessment,
            "confidence_score": clin_result.confidence_score,
        },
        "med": {
            "medications":      list(med_result.medications),
            "status":           med_result.status,
            "source":           med_result.source,
            "reasoning":        med_result.reasoning,
            "assessment":       med_result.assessment,
            "confidence_score": med_result.confidence_score,
        },
        "dx": {
            "matched_codes":        list(dx_result.matched_codes),
            "is_current_diagnosis": dx_result.is_current_diagnosis,
            "reasoning":            dx_result.reasoning,
            "assessment":           dx_result.assessment,
            "confidence_score":     dx_result.confidence_score,
        },
    }
    return pipeline_result, agent_detail, evidence_record
