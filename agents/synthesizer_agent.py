"""
Synthesizer Agent — Senior attending physician final judgment.
Always LLM-based. Receives full clinical note + all 3 specialist reports.
Outputs TWO things: AD/ADRD diagnosis (yes/no) AND subtype classification.

Subtypes: ad | vd | ftd | nsd | na
"""

import json
import re

from logger import get_logger
from schemas import (
    DiagnosisAgentOutput,
    MedicationAgentOutput,
    ClinTextAgentOutput,
    SynthesizerOutput,
)

logger = get_logger(__name__)

# Confidence scores below this threshold trigger Rule B forced downgrade.
# With the new confidence semantics (certainty in own conclusion), 0.40 means
# "uncertain" — a legitimate state, not a weak signal. Only scores < 0.30
# represent agents that are truly unconfident and should not drive a high verdict.
_WEAK_EVIDENCE_THRESHOLD = 0.3

SYNTHESIZER_PROMPT = """You are a senior attending physician and dementia specialist making
the final diagnosis for a patient.

You have received assessments from three specialist colleagues:
1. ClinTextAgent: neurologist who analyzed clinical symptoms
2. MedicationAgent: pharmacist who analyzed medication evidence
3. DiagnosisAgent: coding specialist who analyzed diagnostic codes

You also have access to the complete clinical note.

=== SPECIALIST REPORTS ===

[ClinTextAgent — Neurology]
{clin_report}

[MedicationAgent — Pharmacy]
{med_report}

[DiagnosisAgent — Medical Coding]
{dx_report}

=== COMPLETE CLINICAL NOTE ===
---
{text}
---

=== YOUR TASK ===

Each specialist report includes a rule-based confidence_score (0-1) computed from
objective evidence characteristics, not self-reported by the agent. Use these scores
as reference when weighing evidence, but you retain full clinical judgment authority
to overrule any agent with clear explanation.

PART 1 — AD/ADRD Diagnosis (yes/no):
Synthesize all three specialist reports to determine if this patient
currently has AD/ADRD.

Consider evidence hierarchy:
- Explicit physician documentation of active AD/ADRD (strongest)
- Current AD medications (strong: physician has already diagnosed)
- Current ICD codes with clinical note support (strong)
- Cognitive symptoms documented in current admission (moderate)
- ICD codes without clinical note support (weak: may be historical)
- Single vague mention of cognitive symptoms (weak)

You have full authority to:
- Accept all agents' findings
- Overrule any agent if their reasoning is flawed
- Resolve contradictions between agents with clear explanation

PART 2 — Subtype Classification:
If AD/ADRD is present (final_prediction=1), classify the subtype:
- "ad": Alzheimer's disease
  (progressive memory loss, language problems, gradual onset)
- "vd": Vascular dementia
  (stepwise decline, history of stroke/CVD, focal neurological signs)
- "ftd": Frontotemporal dementia
  (behavioral changes, language problems, younger onset, frontal symptoms)
- "nsd": Non-specific dementia
  (dementia present but insufficient evidence for specific subtype)
- "na": Not applicable (final_prediction=0, no AD/ADRD)

Base subtype on clinical evidence in the note, not just ICD codes.
If AD/ADRD is present but subtype is unclear from the note, use "nsd".

Output ONLY this JSON:
{{
  "final_prediction": 1 or 0,
  "subtype": "ad" or "vd" or "ftd" or "nsd" or "na",
  "confidence": "high" or "medium" or "low",
  "contributing_agents": ["list of agents whose findings supported decision"],
  "causal_chain": "step by step reasoning: what each agent found, how you weighed the evidence, why you reached this conclusion",
  "discrepancy": "describe agent disagreements and how resolved, or null",
  "overruled_agents": ["agent: reason for overruling"],
  "reason": "one sentence final summary"
}}"""


def _parse_llm_json(content: str) -> dict | None:
    content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _format_clin_report(r: ClinTextAgentOutput) -> str:
    ev_str = "\n  - " + "\n  - ".join(r.evidence_list[:5]) if r.evidence_list else "  (none)"
    return (
        f"symptoms_found: {r.symptoms_found} | confidence: {r.confidence} | confidence_score: {r.confidence_score}\n"
        f"Evidence quotes:{ev_str}\n"
        f"Reasoning: {r.reasoning}\n"
        f"Assessment: {r.assessment}"
    )


def _format_med_report(r: MedicationAgentOutput) -> str:
    meds = ", ".join(r.medications) or "(none)"
    return (
        f"meds_found: {r.meds_found} | status: {r.status} | source: {r.source} | confidence: {r.confidence} | confidence_score: {r.confidence_score}\n"
        f"Medications: {meds}\n"
        f"Reasoning: {r.reasoning}\n"
        f"Assessment: {r.assessment}"
    )


def _format_dx_report(r: DiagnosisAgentOutput) -> str:
    codes = ", ".join(r.matched_codes) or "(none)"
    return (
        f"dx_found: {r.dx_found} | is_current_diagnosis: {r.is_current_diagnosis} | confidence: {r.confidence} | confidence_score: {r.confidence_score}\n"
        f"Matched codes: {codes}\n"
        f"Reasoning: {r.reasoning}\n"
        f"Assessment: {r.assessment}"
    )


def _compute_score(
    clin_result: ClinTextAgentOutput,
    med_result:  MedicationAgentOutput,
    dx_result:   DiagnosisAgentOutput,
) -> tuple[int, list[str]]:
    score = 0
    contributing = []
    if clin_result.symptoms_found:
        score += 3
        contributing.append("ClinTextAgent")
    if med_result.meds_found:
        score += 3
        contributing.append("MedicationAgent")
    if dx_result.dx_found:
        score += 1
        contributing.append("DiagnosisAgent")
    return score, contributing


def _fallback_result(
    score:        int,
    contributing: list[str],
    clin_result:  ClinTextAgentOutput,
    med_result:   MedicationAgentOutput,
    dx_result:    DiagnosisAgentOutput,
) -> dict:
    """Rule-based fallback if LLM synthesis fails. Returns a partial dict (no synthesis_mode/scores)."""
    med_found  = med_result.meds_found
    clin_found = clin_result.symptoms_found
    dx_found   = dx_result.dx_found

    if med_found:
        pred, conf = 1, "high"
        reason = "AD medication prescribed — active clinical management confirms diagnosis."
    elif clin_found and dx_found:
        pred, conf = 1, "medium"
        reason = "Clinical symptoms and ICD codes both positive."
    elif clin_found:
        pred, conf = 1, "low"
        reason = "Clinical symptoms positive, no corroboration — low confidence."
    elif dx_found:
        pred, conf = 1, "low"
        reason = "ICD codes only — may be historical coding."
    else:
        pred, conf = 0, "high"
        reason = "All agents negative — no AD/ADRD evidence."

    return {
        "final_prediction":    pred,
        "subtype":             "nsd" if pred == 1 else "na",
        "confidence":          conf,
        "total_score":         score,
        "contributing_agents": contributing,
        "causal_chain":        f"[Fallback — LLM failed] {reason}",
        "discrepancy":         None,
        "overruled_agents":    [],
        "reason":              reason,
    }


def _apply_confidence_correction(
    result:      dict,
    clin_result: ClinTextAgentOutput,
    med_result:  MedicationAgentOutput,
    dx_result:   DiagnosisAgentOutput,
) -> dict:
    """Post-processing rules that prevent the Synthesizer from being over-confident.

    Rules stack — all applicable rules are applied in order before returning.
    """
    current_confidence = result.get("confidence", "low")
    final_prediction   = result.get("final_prediction", 0)
    overruled          = result.get("overruled_agents", [])
    contributing       = result.get("contributing_agents", [])

    score_map = {
        "ClinTextAgent":   clin_result.confidence_score,
        "MedicationAgent": med_result.confidence_score,
        "DiagnosisAgent":  dx_result.confidence_score,
    }
    contributing_scores = [score_map[a] for a in contributing if a in score_map]

    correction_reasons: list[str] = []

    # Rule A: overruled agents present → cap at medium
    if overruled and current_confidence == "high":
        result["confidence"] = "medium"
        current_confidence   = "medium"
        correction_reasons.append("overruled agents present")

    # Rule B: all contributing agents have weak evidence → cap at low
    if (
        contributing_scores
        and all(s < _WEAK_EVIDENCE_THRESHOLD for s in contributing_scores)
        and current_confidence in ("high", "medium")
    ):
        result["confidence"] = "low"
        current_confidence   = "low"
        correction_reasons.append("all contributing agents have weak evidence")

    # Rule C: positive prediction with no contributing agents → force low
    if final_prediction == 1 and not contributing:
        result["confidence"] = "low"
        current_confidence   = "low"  # noqa: F841
        correction_reasons.append("positive prediction with no contributing agents")

    if correction_reasons:
        result["confidence_correction"] = "Downgraded: " + " | ".join(correction_reasons)
    else:
        result["confidence_correction"] = result.get("confidence_correction") or None

    return result


def run_synthesizer_agent(
    dx_result:   DiagnosisAgentOutput,
    med_result:  MedicationAgentOutput,
    clin_result: ClinTextAgentOutput,
    llm,
    text:        str = "",
) -> SynthesizerOutput:
    """
    Args:
        dx_result:   output from diagnosis_agent   (ICD corroboration)
        med_result:  output from medication_agent  (pharmacological confirmation)
        clin_result: output from clintext_agent    (primary clinical evidence)
        llm:         LangChain LLM instance
        text:        full clinical note text
    Returns:
        SynthesizerOutput validated instance
    """
    score, contributing = _compute_score(clin_result, med_result, dx_result)

    # Extract per-agent rule-based confidence scores
    cs_clin = clin_result.confidence_score
    cs_med  = med_result.confidence_score
    cs_dx   = dx_result.confidence_score

    # ── Consensus-based early exit ────────────────────────────────────────────
    med_pos  = med_result.meds_found
    clin_pos = clin_result.symptoms_found
    dx_pos   = dx_result.dx_found

    if med_pos and clin_pos and dx_pos:
        consensus_result = SynthesizerOutput(
            final_prediction=1,
            subtype="nsd",
            confidence="high",
            synthesis_mode="consensus_positive",
            total_score=score,
            contributing_agents=contributing,
            causal_chain=(
                "All specialist agents reached consensus: MedicationAgent, ClinTextAgent, "
                "and DiagnosisAgent all positive. LLM arbitration skipped."
            ),
            discrepancy=None,
            overruled_agents=[],
            reason="Consensus positive — medication and clinical evidence both confirmed.",
            confidence_score_clin=cs_clin,
            confidence_score_med=cs_med,
            confidence_score_dx=cs_dx,
            confidence_correction=None,
        )
        corrected = _apply_confidence_correction(
            consensus_result.model_dump(),
            clin_result,
            med_result,
            dx_result,
        )
        return SynthesizerOutput(**corrected)

    if not med_pos and not clin_pos and not dx_pos:
        return SynthesizerOutput(
            final_prediction=0,
            subtype="na",
            confidence="high",
            synthesis_mode="consensus_negative",
            total_score=score,
            contributing_agents=[],
            causal_chain=(
                "All specialist agents reached consensus: no AD/ADRD evidence found. "
                "LLM arbitration skipped."
            ),
            discrepancy=None,
            overruled_agents=[],
            reason="Consensus negative — no medication, symptom, or ICD evidence found.",
            confidence_score_clin=cs_clin,
            confidence_score_med=cs_med,
            confidence_score_dx=cs_dx,
            confidence_correction=None,
        )

    # ── LLM arbitration (mixed signals) ──────────────────────────────────────

    prompt = SYNTHESIZER_PROMPT.format(
        clin_report=_format_clin_report(clin_result),
        med_report=_format_med_report(med_result),
        dx_report=_format_dx_report(dx_result),
        text=text,
    )

    try:
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        parsed   = _parse_llm_json(content)

        if parsed:
            pred       = int(parsed.get("final_prediction", 0))
            subtype_raw = str(parsed.get("subtype", "na")).lower().strip()
            if subtype_raw not in ("ad", "vd", "ftd", "nsd", "na"):
                subtype_raw = "nsd" if pred == 1 else "na"
            if pred == 0:
                subtype_raw = "na"

            conf_raw = str(parsed.get("confidence", "low")).lower().strip()
            conf     = conf_raw if conf_raw in ("high", "medium", "low") else "low"

            result = {
                "final_prediction":      pred,
                "subtype":               subtype_raw,
                "confidence":            conf,
                "synthesis_mode":        "llm_arbitration",
                "total_score":           score,
                "contributing_agents":   parsed.get("contributing_agents", contributing),
                "causal_chain":          str(parsed.get("causal_chain", "")),
                "discrepancy":           parsed.get("discrepancy") or None,
                "overruled_agents":      parsed.get("overruled_agents", []),
                "reason":                str(parsed.get("reason", "")),
                "confidence_score_clin": cs_clin,
                "confidence_score_med":  cs_med,
                "confidence_score_dx":   cs_dx,
                "confidence_correction": None,
            }
            # Post-processing: prevent over-confident LLM outputs
            result = _apply_confidence_correction(result, clin_result, med_result, dx_result)
            return SynthesizerOutput(**result)

    except Exception:
        logger.warning("Synthesizer LLM call failed, using fallback")

    # ── Deterministic fallback (LLM failed) ──────────────────────────────────
    fallback = _fallback_result(score, contributing, clin_result, med_result, dx_result)
    return SynthesizerOutput(
        **fallback,
        synthesis_mode="llm_arbitration",
        confidence_score_clin=cs_clin,
        confidence_score_med=cs_med,
        confidence_score_dx=cs_dx,
        confidence_correction=None,
    )
