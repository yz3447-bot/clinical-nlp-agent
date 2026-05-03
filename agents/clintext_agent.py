"""
Clinical Text Agent — primary evidence layer.
Neurologist reads the FULL clinical note for chronic AD/ADRD evidence.

Two-phase execution when a boundary signal is detected:
  Phase 1: standard LLM call → initial judgment
  Phase 2: if boundary signal present, retrieve relevant principles from
           BoundaryKnowledgeBase and re-evaluate with expert guidance injected.
"""

import json
import re

from logger import get_logger
from schemas import ClinTextAgentOutput

logger = get_logger(__name__)

CLINTEXT_PROMPT = """You are an experienced clinical neurologist specializing in dementia diagnosis.

Your task is to analyze the complete clinical note and determine whether
this patient has chronic AD/ADRD, drawing on ALL documentation in the note
including past medical history, current admission findings, and medications.

As an experienced neurologist, you understand that:
- Family history of dementia does not constitute a patient diagnosis
- Ruled-out differential diagnoses are not active conditions
- A prior chronic diagnosis of AD/ADRD documented in PMH IS valid evidence —
  a single mention of "dementia" or "Alzheimer's" in past medical history
  counts as supporting evidence for a chronic neurodegenerative condition
- Acute delirium (from infection, metabolic causes, post-operative state,
  or medication) is distinct from chronic AD/ADRD and does NOT count
  as evidence of chronic dementia on its own
- Acute metabolic encephalopathy or post-operative confusion are NOT
  evidence of underlying chronic dementia unless the note also documents
  a pre-existing cognitive condition

Focus on:
- Past Medical History (PMH) — prior dementia diagnoses are valid evidence
- Chief Complaint and History of Present Illness
- Physical and neurological examination findings
- Assessment & Plan written by the attending physician
- Discharge diagnosis and discharge summary
- Any cognitive assessments or behavioral observations

For each piece of evidence, quote the EXACT text from the note.
If you find no evidence of chronic AD/ADRD, say so clearly.

Clinical note:
---
{text}
---

Output ONLY this JSON:
{{
  "symptoms_found": true or false,
  "confidence": "high" or "medium" or "low",
  "evidence_list": [
    {{"text": "exact quote from note", "strength": "strong" or "moderate" or "weak"}}
  ],
  "hedging_detected": true or false,
  "acute_only": true or false,
  "reasoning": "explain why you included or excluded key findings",
  "assessment": "one sentence: your clinical impression of this patient"
}}

hedging_detected: set true if the ONLY positive evidence uses uncertain language
like "possible dementia", "suspected cognitive impairment", "rule out dementia",
or "recommend neurocognitive evaluation". Set false if there is any definitive
statement of dementia/AD/ADRD.

acute_only: set true if all cognitive symptoms are attributed to acute reversible
causes (infection, metabolic, post-operative, medication) AND the note contains
no mention of chronic pre-existing dementia. Set false if there is any chronic
cognitive history, even in PMH.

Evidence strength:
- strong: physician explicitly states AD/ADRD diagnosis, or cognitive assessment
  confirms dementia, or patient is on AD medications
- moderate: documented cognitive impairment with clinical specificity
  (e.g. "significant memory loss", "oriented to person only", "MMSE 18/30")
- weak: vague or indirect mention (e.g. "some confusion", "forgetful at baseline")

Confidence levels:
- high: you are highly certain about your conclusion (positive or negative)
- medium: some supporting or refuting evidence but with gaps or ambiguity
- low: minimal evidence; uncertain judgment"""


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


def _compute_confidence_score(
    symptoms_found: bool,
    evidence_list: list,
    hedging_detected: bool = False,
    acute_only: bool = False,
) -> float:
    """Confidence = certainty in ClinTextAgent's own conclusion (positive or negative).

    symptoms_found=False, acute_only=True  → 0.50  (cogn. symptoms exist but reversible;
                                                      cannot rule out underlying chronic)
    symptoms_found=False, otherwise        → 0.75  (no cognitive evidence found)
    symptoms_found=True,  hedging_detected → 0.35  (only uncertain language)
    symptoms_found=True,  strong ≥ 2       → 0.90
    symptoms_found=True,  strong = 1       → 0.75
    symptoms_found=True,  total ≥ 2        → 0.60
    symptoms_found=True,  otherwise        → 0.40
    """
    if not symptoms_found:
        return 0.50 if acute_only else 0.75

    if not evidence_list:
        return 0.35

    if hedging_detected:
        return 0.35

    # Support both new dict format and legacy plain-string format
    if evidence_list and isinstance(evidence_list[0], dict):
        strong = sum(1 for e in evidence_list if e.get("strength") == "strong")
        total  = len(evidence_list)
    else:
        strong = 0
        total  = len(evidence_list)

    if strong >= 2:
        return 0.90
    if strong >= 1:
        return 0.75
    if total >= 2:
        return 0.60
    return 0.40


def _build_boundary_context(principles: list[dict]) -> str:
    lines = [
        "=== BOUNDARY JUDGMENT PRINCIPLES ===",
        "This case has been flagged as a boundary case.",
        "",
    ]
    for i, p in enumerate(principles, 1):
        lines += [
            f"Principle {i}: [{p['boundary_type']}]",
            f"Clinical features: {p['clinical_features']}",
            f"Judgment principle: {p['principle']}",
            f"Key evidence to look for: {p['key_evidence']}",
            f"Correct conclusion: {p['correct_conclusion']}",
            f"Common mistake to avoid: {p['common_mistake']}",
            "",
        ]
    lines.append("=== RE-EVALUATE THE CLINICAL NOTE WITH ABOVE PRINCIPLES ===")
    return "\n".join(lines)


def run_clintext_agent(row: dict, llm, boundary_kb=None) -> ClinTextAgentOutput:
    """
    Args:
        row:         dict with 'text' key (full clinical note)
        llm:         LangChain LLM instance
        boundary_kb: optional BoundaryKnowledgeBase; triggers a second LLM call
                     with retrieved principles when a boundary signal is detected
    Returns:
        ClinTextAgentOutput validated instance
    """
    text = str(row.get("text", "") or "")

    if not text.strip():
        return ClinTextAgentOutput(
            symptoms_found=False,
            evidence_list=[],
            reasoning="No clinical text provided.",
            assessment="No clinical text available.",
            confidence="low",
            confidence_score=0.0,
        )

    prompt = CLINTEXT_PROMPT.format(text=text)

    try:
        # ── Phase 1: standard LLM call ────────────────────────────────────────
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        parsed   = _parse_llm_json(content)

        if not parsed:
            raise ValueError("Unparseable LLM response")

        found            = bool(parsed.get("symptoms_found", False))
        conf_raw         = str(parsed.get("confidence", "low")).lower().strip()
        conf             = conf_raw if conf_raw in ("high", "medium", "low") else "low"
        evidence_raw     = parsed.get("evidence_list", [])
        hedging_detected = bool(parsed.get("hedging_detected", False))
        acute_only       = bool(parsed.get("acute_only", False))

        boundary_principles_applied: list[str] = []

        # ── Phase 2: boundary-aware refinement ────────────────────────────────
        should_refine = (
            hedging_detected
            or acute_only
            or (found and conf == "low")
            or (not found and len(evidence_raw) > 0)
        )

        if should_refine and boundary_kb is not None:
            features: list[str] = []
            if hedging_detected:
                features.append("uncertain language possible suspected dementia")
            if acute_only:
                features.append("acute cognitive symptoms only discharge normal")
            if found and conf == "low":
                features.append("weak evidence uncertain conclusion")
            if not found and evidence_raw:
                features.append("contradiction symptoms not found but evidence present")

            query = " | ".join(features) if features else "boundary case uncertain judgment"

            try:
                principles = boundary_kb.retrieve(query, n_results=2)
                if principles:
                    boundary_principles_applied = [p["boundary_type"] for p in principles]
                    boundary_context = _build_boundary_context(principles)
                    refined_prompt   = boundary_context + "\n\n" + prompt
                    second_response  = llm.invoke(refined_prompt)
                    second_content   = (
                        second_response.content
                        if hasattr(second_response, "content")
                        else str(second_response)
                    )
                    second_parsed = _parse_llm_json(second_content)
                    if second_parsed:
                        parsed = second_parsed   # use Phase 2 result
                        found            = bool(parsed.get("symptoms_found", False))
                        conf_raw         = str(parsed.get("confidence", "low")).lower().strip()
                        conf             = conf_raw if conf_raw in ("high", "medium", "low") else "low"
                        evidence_raw     = parsed.get("evidence_list", [])
                        hedging_detected = bool(parsed.get("hedging_detected", False))
                        acute_only       = bool(parsed.get("acute_only", False))
                        logger.info(
                            "  [ClinTextAgent] Boundary refinement applied: %s → symptoms_found=%s conf=%s",
                            boundary_principles_applied, found, conf,
                        )
                    else:
                        logger.warning("  [ClinTextAgent] Phase 2 parse failed — keeping Phase 1 result")
            except Exception as e:
                logger.warning("  [ClinTextAgent] Boundary retrieval/refinement failed: %s", e)

        # Normalize evidence to plain strings for schema; keep raw dicts for scoring
        evidence = [
            e["text"] if isinstance(e, dict) else str(e)
            for e in evidence_raw
        ]

        if found and not evidence:
            conf = "low"

        return ClinTextAgentOutput(
            symptoms_found=found,
            evidence_list=evidence,
            reasoning=str(parsed.get("reasoning", "")),
            assessment=str(parsed.get("assessment", "")),
            confidence=conf,
            confidence_score=_compute_confidence_score(
                found, evidence_raw, hedging_detected, acute_only
            ),
            boundary_principles_applied=boundary_principles_applied,
        )

    except Exception as e:
        logger.warning(f"ClinTextAgent LLM call failed: {e}")
        return ClinTextAgentOutput(
            symptoms_found=False,
            evidence_list=[],
            reasoning=f"Agent error: {e}",
            assessment="Agent error — defaulting negative.",
            confidence="low",
            confidence_score=0.0,
        )
