"""
Synthesizer Agent — Senior attending physician final judgment.
Always LLM-based. Receives full clinical note + all 3 specialist reports.
Outputs TWO things: AD/ADRD diagnosis (yes/no) AND subtype classification.

Subtypes: ad | vd | ftd | nsd | na
"""

import json
import re

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


def _format_clin_report(r: dict) -> str:
    found      = r.get("symptoms_found", False)
    conf       = r.get("confidence", "low")
    cs         = r.get("confidence_score", 0.0)
    ev         = r.get("evidence_list", [])
    ev_str     = "\n  - " + "\n  - ".join(ev[:5]) if ev else "  (none)"
    reasoning  = r.get("reasoning", "")
    assessment = r.get("assessment", "")
    return (
        f"symptoms_found: {found} | confidence: {conf} | confidence_score: {cs}\n"
        f"Evidence quotes:{ev_str}\n"
        f"Reasoning: {reasoning}\n"
        f"Assessment: {assessment}"
    )


def _format_med_report(r: dict) -> str:
    found      = r.get("meds_found", False)
    meds       = ", ".join(r.get("medications", [])) or "(none)"
    status     = r.get("status", "none")
    source     = r.get("source", "none")
    conf       = r.get("confidence", "low")
    cs         = r.get("confidence_score", 0.0)
    reasoning  = r.get("reasoning", "")
    assessment = r.get("assessment", "")
    return (
        f"meds_found: {found} | status: {status} | source: {source} | confidence: {conf} | confidence_score: {cs}\n"
        f"Medications: {meds}\n"
        f"Reasoning: {reasoning}\n"
        f"Assessment: {assessment}"
    )


def _format_dx_report(r: dict) -> str:
    found      = r.get("dx_found", False)
    codes      = ", ".join(r.get("matched_codes", [])) or "(none)"
    current    = r.get("is_current_diagnosis", False)
    conf       = r.get("confidence", "low")
    cs         = r.get("confidence_score", 0.0)
    reasoning  = r.get("reasoning", "")
    assessment = r.get("assessment", "")
    return (
        f"dx_found: {found} | is_current_diagnosis: {current} | confidence: {conf} | confidence_score: {cs}\n"
        f"Matched codes: {codes}\n"
        f"Reasoning: {reasoning}\n"
        f"Assessment: {assessment}"
    )


def _compute_score(clin_result: dict, med_result: dict, dx_result: dict) -> tuple[int, list[str]]:
    score = 0
    contributing = []
    if clin_result.get("symptoms_found"):
        score += 3
        contributing.append("ClinTextAgent")
    if med_result.get("meds_found"):
        score += 3
        contributing.append("MedicationAgent")
    if dx_result.get("dx_found"):
        score += 1
        contributing.append("DiagnosisAgent")
    return score, contributing


def _fallback_result(score: int, contributing: list, clin_result: dict,
                     med_result: dict, dx_result: dict) -> dict:
    """Rule-based fallback if LLM synthesis fails."""
    med_found  = med_result.get("meds_found", False)
    clin_found = clin_result.get("symptoms_found", False)
    dx_found   = dx_result.get("dx_found", False)

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
    result: dict,
    clin_result: dict,
    med_result: dict,
    dx_result: dict,
) -> dict:
    """Post-processing rules that prevent the Synthesizer from being over-confident."""
    clin_score = clin_result.get("confidence_score", 0.0)
    med_score  = med_result.get("confidence_score", 0.0)
    dx_score   = dx_result.get("confidence_score", 0.0)

    current_confidence  = result.get("confidence", "low")
    has_discrepancy     = bool(result.get("discrepancy"))
    has_overruled       = bool(result.get("overruled_agents"))
    low_evidence_count  = sum(1 for s in (clin_score, med_score, dx_score) if s < 0.3)
    sparse_early_case   = (
        med_score == 0.0 and
        len(clin_result.get("evidence_list", [])) < 2
    )

    # Rule 1: discrepancy + overruled agents → cap at medium
    if has_discrepancy and has_overruled:
        if current_confidence == "high":
            result["confidence"] = "medium"
            result["confidence_correction"] = "Downgraded: discrepancy with overruled agents"
            return result

    # Rule 2: majority of agents have weak evidence → cap at low
    if low_evidence_count >= 2:
        if current_confidence in ("high", "medium"):
            result["confidence"] = "low"
            result["confidence_correction"] = "Downgraded: insufficient evidence across agents"
            return result

    # Rule 3: no medication evidence + sparse clinical quotes → cap at low
    if sparse_early_case:
        if current_confidence == "high":
            result["confidence"] = "low"
            result["confidence_correction"] = "Downgraded: sparse evidence, possible early-stage case"
            return result

    return result


def run_synthesizer_agent(
    dx_result:   dict,
    med_result:  dict,
    clin_result: dict,
    llm,
    text:        str = "",
) -> dict:
    """
    Args:
        dx_result:   output from diagnosis_agent   (weight 1)
        med_result:  output from medication_agent  (weight 3)
        clin_result: output from clintext_agent    (weight 3)
        llm:         LangChain LLM instance
        text:        full clinical note text
    Returns:
        {
          "final_prediction": int,
          "subtype": str,
          "confidence": str,
          "synthesis_mode": str,        # "consensus_positive" | "consensus_negative" | "llm_arbitration"
          "total_score": int,
          "contributing_agents": list[str],
          "causal_chain": str,
          "discrepancy": str | None,
          "overruled_agents": list[str],
          "reason": str,
          "confidence_score_clin": float,
          "confidence_score_med": float,
          "confidence_score_dx": float,
          "confidence_correction": str | None
        }
    """
    score, contributing = _compute_score(clin_result, med_result, dx_result)

    # Extract per-agent rule-based confidence scores
    cs_clin = float(clin_result.get("confidence_score", 0.0))
    cs_med  = float(med_result.get("confidence_score", 0.0))
    cs_dx   = float(dx_result.get("confidence_score", 0.0))

    # ── Consensus-based early exit ────────────────────────────────────────────
    med_pos  = bool(med_result.get("meds_found", False))
    clin_pos = bool(clin_result.get("symptoms_found", False))
    dx_pos   = bool(dx_result.get("dx_found", False))

    if med_pos and clin_pos:
        return {
            "final_prediction":      1,
            "subtype":               "nsd",
            "confidence":            "high",
            "synthesis_mode":        "consensus_positive",
            "total_score":           score,
            "contributing_agents":   contributing,
            "causal_chain":          (
                "All specialist agents reached consensus: MedicationAgent and ClinTextAgent "
                "both positive. LLM arbitration skipped."
            ),
            "discrepancy":           None,
            "overruled_agents":      [],
            "reason":                "Consensus positive — medication and clinical evidence both confirmed.",
            "confidence_score_clin": cs_clin,
            "confidence_score_med":  cs_med,
            "confidence_score_dx":   cs_dx,
            "confidence_correction": None,
        }

    if not med_pos and not clin_pos and not dx_pos:
        return {
            "final_prediction":      0,
            "subtype":               "na",
            "confidence":            "high",
            "synthesis_mode":        "consensus_negative",
            "total_score":           score,
            "contributing_agents":   [],
            "causal_chain":          (
                "All specialist agents reached consensus: no AD/ADRD evidence found. "
                "LLM arbitration skipped."
            ),
            "discrepancy":           None,
            "overruled_agents":      [],
            "reason":                "Consensus negative — no medication, symptom, or ICD evidence found.",
            "confidence_score_clin": cs_clin,
            "confidence_score_med":  cs_med,
            "confidence_score_dx":   cs_dx,
            "confidence_correction": None,
        }

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
            pred    = int(parsed.get("final_prediction", 0))
            subtype = str(parsed.get("subtype", "na")).lower().strip()
            if subtype not in ("ad", "vd", "ftd", "nsd", "na"):
                subtype = "nsd" if pred == 1 else "na"
            if pred == 0:
                subtype = "na"

            result = {
                "final_prediction":      pred,
                "subtype":               subtype,
                "confidence":            str(parsed.get("confidence", "low")),
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
            return result
    except Exception:
        pass

    result = _fallback_result(score, contributing, clin_result, med_result, dx_result)
    result.update({
        "synthesis_mode":        "llm_arbitration",
        "confidence_score_clin": cs_clin,
        "confidence_score_med":  cs_med,
        "confidence_score_dx":   cs_dx,
        "confidence_correction": None,
    })
    return result
