"""
Clinical Text Agent — Layer 1 (primary evidence, weight +3).
Neurologist reads the FULL clinical note for current AD/ADRD evidence.
Returns exact quotes, explicit reasoning, and exclusion list.
"""

import json
import re

from logger import get_logger
from schemas import ClinTextAgentOutput

logger = get_logger(__name__)

CLINTEXT_PROMPT = """You are an experienced clinical neurologist specializing in dementia diagnosis.

Your task is to analyze the complete clinical note and identify evidence
of CURRENT AD/ADRD based solely on this admission.

As an experienced neurologist, you understand that:
- Family history of dementia does not constitute a patient diagnosis
- Ruled-out differential diagnoses are not active conditions
- Delirium or acute confusion from infection/metabolic causes is
  distinct from AD/ADRD
- A historical diagnosis carried forward in PMH without current
  clinical management is different from an active diagnosis

Focus on:
- Chief Complaint and History of Present Illness
- Physical and neurological examination findings
- Assessment & Plan written by the attending physician
- Discharge diagnosis and discharge summary
- Any cognitive assessments or behavioral observations

For each piece of evidence, quote the EXACT text from the note.
If you find no current evidence, say so clearly.

Clinical note:
---
{text}
---

Output ONLY this JSON:
{{
  "symptoms_found": true or false,
  "confidence": "high" or "medium" or "low",
  "evidence_list": ["exact quote from note 1", "exact quote 2"],
  "reasoning": "explain why you included or excluded key findings",
  "assessment": "one sentence: your clinical impression of this patient"
}}

Confidence levels:
- high: physician explicitly documents active AD/ADRD diagnosis OR
        2+ distinct categories of current cognitive symptoms with quotes
- medium: implicit cognitive symptoms without explicit diagnosis
- low: single vague mention or ambiguous finding"""


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


def _compute_confidence_score(symptoms_found: bool, evidence_list: list[str]) -> float:
    """Rule-based confidence score derived from objective evidence characteristics."""
    if not symptoms_found or not evidence_list:
        return 0.0

    score = 0.0
    n = len(evidence_list)

    # Evidence quantity (four tiers)
    if n >= 5:
        score += 0.5
    elif n >= 2:
        score += 0.35
    else:
        score += 0.15

    # Source quality
    ev_text = " ".join(evidence_list).lower()
    if any(kw in ev_text for kw in ("brief hospital course", "assessment", "plan", "discharge")):
        score += 0.3
    elif any(kw in ev_text for kw in ("hpi",)):
        score += 0.2
    elif any(kw in ev_text for kw in ("pmh", "past medical")):
        score += 0.05

    # Negation penalty: only if majority of quotes contain negation
    negations = ("no evidence", "ruled out", "no dementia", "no cognitive")
    negated_count = sum(
        1 for quote in evidence_list
        if any(phrase in quote.lower() for phrase in negations)
    )
    if negated_count > n / 2:
        score -= 0.2

    return round(max(0.0, min(1.0, score)), 4)


def _format_rag_context(cases: list[dict]) -> str:
    """Format retrieved cases into a prompt-ready reference block."""
    lines = ["Reference cases from knowledge base:"]
    for i, case in enumerate(cases, 1):
        if case["label"] == 1:
            descriptor = f"AD present, subtype: {case['subtype']}"
        else:
            descriptor = "No AD"
        lines.append(f"Case {i} ({descriptor}): {case['text']}")
    return "\n".join(lines)


def run_clintext_agent(row: dict, llm, kb=None) -> ClinTextAgentOutput:
    """
    Args:
        row: dict with 'text' key (full clinical note)
        llm: LangChain LLM instance
        kb:  optional KnowledgeBase instance; if provided, retrieves 3
             similar cases and prepends them to the prompt as context
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

    # ── Optionally prepend RAG context ────────────────────────────────────────
    prompt = CLINTEXT_PROMPT.format(text=text)
    if kb is not None:
        try:
            similar_cases = kb.retrieve_similar(text, n_results=3)
            if similar_cases:
                rag_block = _format_rag_context(similar_cases)
                prompt = rag_block + "\n\n" + prompt
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")

    try:
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        parsed   = _parse_llm_json(content)

        if not parsed:
            raise ValueError("Unparseable LLM response")

        found      = bool(parsed.get("symptoms_found", False))
        conf_raw   = str(parsed.get("confidence", "low")).lower().strip()
        conf       = conf_raw if conf_raw in ("high", "medium", "low") else "low"
        evidence   = parsed.get("evidence_list", [])
        reasoning  = str(parsed.get("reasoning", ""))
        assessment = str(parsed.get("assessment", ""))

        if found and not evidence:
            conf = "low"

        return ClinTextAgentOutput(
            symptoms_found=found,
            evidence_list=evidence,
            reasoning=reasoning,
            assessment=assessment,
            confidence=conf,
            confidence_score=_compute_confidence_score(found, evidence),
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
