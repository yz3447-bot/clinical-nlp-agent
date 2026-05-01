"""
Medication Agent — Layer 2 (pharmacological confirmation, weight +3).
Phase 1: human annotation column (fast path).
Phase 2: LLM free-text scan of medication sections.
"""

import json
import re

from logger import get_logger
from schemas import MedicationAgentOutput

logger = get_logger(__name__)


def _extract_med_sections(text: str) -> str:
    """
    Extract medication-related sections from a clinical note.
    Targets: Discharge Medications, Medications on Admission/Discharge,
             Medication Reconciliation, Home Medications.
    Falls back to the first 2000 chars when no section is found.
    """
    patterns = [
        r"(?i)(discharge\s+medications?.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
        r"(?i)(medications?\s+on\s+(?:admission|discharge).*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
        r"(?i)(medication\s+reconciliation.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
        r"(?i)(home\s+medications?.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
    ]

    extracted = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        extracted.extend(m.strip() for m in matches if m.strip())

    if extracted:
        combined = "\n\n".join(dict.fromkeys(extracted))
        return combined[:3000]

    return text[:2000]


_ANNOTATION_MED_KEYWORDS = frozenset({
    "donepezil", "aricept",
    "memantine", "namenda",
    "rivastigmine", "exelon",
    "galantamine", "razadyne",
    "tacrine",
})

MED_TEXT_PROMPT = """You are a clinical pharmacist specializing in dementia care.

Your task is to identify AD/ADRD medications in this clinical note.
Note: structured medication columns in this dataset are unreliable,
so focus entirely on the free-text clinical note.

AD/ADRD medications to identify:
- Donepezil (Aricept)
- Memantine (Namenda)
- Rivastigmine (Exelon)
- Galantamine (Razadyne)
- Tacrine
- Any cholinesterase inhibitor prescribed for cognitive symptoms

As an experienced pharmacist, you know the critical distinction between:
- CURRENTLY PRESCRIBED: patient is actively taking this medication
  during this admission or at discharge
- HISTORICAL: patient took this in the past but discontinued
- REFUSED: medication was offered but patient declined
- MENTIONED: discussed in context of diagnosis but not prescribed
- NONE: no AD medication found

Only set meds_found=true for CURRENTLY PRESCRIBED status.
Look for evidence in: medication reconciliation, discharge medications,
home medications, and medication orders.

IMPORTANT: Medications documented in "Medications on Admission" or home
medication lists count as valid current evidence, even if this admission
is for an unrelated condition. For example, a patient taking donepezil at
home who is admitted for pneumonia should still be classified as
meds_found=true, status=current. Do not classify admission medications as
"historical" simply because the patient was admitted for another reason.

Clinical note:
---
{text}
---

Output ONLY this JSON:
{{
  "meds_found": true or false,
  "medications": ["drug name and dose if available"],
  "status": "current" or "historical" or "refused" or "mentioned" or "none",
  "confidence": "high" or "medium" or "low",
  "reasoning": "explain what you found and why you classified it this way",
  "assessment": "one sentence: medication evidence for or against AD/ADRD"
}}"""


def _compute_confidence_score(
    meds_found: bool,
    source: str,
    status: str,
    medications: list[str] | None = None,
) -> float:
    """Rule-based confidence score derived from medication source and status."""
    if not meds_found:
        # Weak negative: absence of AD medications is not conclusive.
        # Many confirmed AD/ADRD patients are not prescribed these medications.
        return 0.30
    if source in ("annotation", "structured"):
        return 0.95
    if status == "current":
        meds_lower = " ".join(m.lower() for m in (medications or []))
        if any(kw in meds_lower for kw in _ANNOTATION_MED_KEYWORDS):
            return 0.75
        return 0.75
    if status in ("historical", "refused"):
        return 0.55
    if status == "mentioned":
        return 0.05
    return 0.30


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


def run_medication_agent(row: dict, llm=None) -> MedicationAgentOutput:
    """
    Args:
        row: dict with 'ad_med_annotation' (human-annotated med text) and 'text' key
        llm: LangChain LLM instance
    Returns:
        MedicationAgentOutput validated instance
    """
    # ── Phase 1: human annotation column (fast path) ─────────────────────────
    annotation = str(row.get("ad_med_annotation", "") or "").lower()
    if annotation.strip():
        found_meds = [kw for kw in _ANNOTATION_MED_KEYWORDS if kw in annotation]
        if found_meds:
            return MedicationAgentOutput(
                meds_found=True,
                medications=found_meds,
                status="current",
                source="annotation",
                confidence="high",
                confidence_score=_compute_confidence_score(True, "annotation", "current"),
                reasoning=f"Human annotation confirms AD medication(s): {', '.join(found_meds)}",
                assessment=f"Active AD medication confirmed via human annotation: {', '.join(found_meds)}",
            )

    # ── Phase 2: LLM full-text scan ───────────────────────────────────────────
    text = str(row.get("text", "") or "")
    if not text.strip() or llm is None:
        return MedicationAgentOutput(
            meds_found=False,
            medications=[],
            status="none",
            source="none",
            confidence="low",
            confidence_score=_compute_confidence_score(False, "none", "none"),
            reasoning="No text available for LLM scan.",
            assessment="Cannot assess — no clinical text.",
        )

    med_text = _extract_med_sections(text)
    used_extraction = len(med_text) < len(text)

    try:
        response = llm.invoke(MED_TEXT_PROMPT.format(text=med_text))
        content  = response.content if hasattr(response, "content") else str(response)
        parsed   = _parse_llm_json(content)

        if parsed:
            found      = bool(parsed.get("meds_found", False))
            status_raw = str(parsed.get("status", "none")).lower().strip()
            status     = (status_raw if status_raw in
                          ("current", "historical", "refused", "mentioned", "none")
                          else "none")
            conf_raw   = str(parsed.get("confidence", "low")).lower().strip()
            conf       = conf_raw if conf_raw in ("high", "medium", "low") else "low"
            prefix = "[Section extracted] " if used_extraction else "[Full text] "
            return MedicationAgentOutput(
                meds_found=found,
                medications=parsed.get("medications", []),
                status=status,
                source="text",
                confidence=conf,
                confidence_score=_compute_confidence_score(found, "text", status, parsed.get("medications", [])),
                reasoning=prefix + str(parsed.get("reasoning", "")),
                assessment=str(parsed.get("assessment", "")),
            )
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")

    return MedicationAgentOutput(
        meds_found=False,
        medications=[],
        status="none",
        source="none",
        confidence="low",
        confidence_score=_compute_confidence_score(False, "none", "none"),
        reasoning="LLM scan failed.",
        assessment="LLM scan failed — defaulting negative.",
    )
