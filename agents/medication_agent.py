"""
Medication Agent — Layer 2 (pharmacological confirmation, weight +3).
Clinical pharmacist reads the FULL clinical note for AD/ADRD medications.
Structured columns are unreliable in this dataset (all zeros) — free text is primary.
"""

import json
import re

from logger import get_logger
from schemas import MedicationAgentOutput

logger = get_logger(__name__)

# Structured column names to check first (Phase 1 fast path)
AD_MED_COLUMNS = [
    "donepezil", "denepezil",
    "memantine",
    "tacrine",
    "rivastigmine",
    "galantamine",
]

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


def _compute_confidence_score(meds_found: bool, source: str, status: str) -> float:
    """Rule-based confidence score derived from medication source and status."""
    if not meds_found:
        return 0.0
    if source == "structured":
        return 0.9
    if status == "current":
        return 0.7
    if status == "historical":
        return 0.2
    if status in ("refused", "mentioned"):
        return 0.1
    return 0.0


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
        row: dict with medication flag columns and 'text' key (full clinical note)
        llm: LangChain LLM instance
    Returns:
        MedicationAgentOutput validated instance
    """
    # ── Phase 1: structured binary columns (fast path) ────────────────────────
    structured_meds = []
    for col in AD_MED_COLUMNS:
        if col not in row:
            continue
        try:
            if int(float(str(row[col]))) == 1:
                canonical = "donepezil" if col == "denepezil" else col
                if canonical not in structured_meds:
                    structured_meds.append(canonical)
        except (ValueError, TypeError):
            pass

    if structured_meds:
        return MedicationAgentOutput(
            meds_found=True,
            medications=structured_meds,
            status="current",
            source="structured",
            confidence="high",
            confidence_score=_compute_confidence_score(True, "structured", "current"),
            reasoning=f"Structured columns flag: {', '.join(structured_meds)}.",
            assessment=f"Active prescription confirmed via structured data: {', '.join(structured_meds)}.",
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
            confidence_score=0.0,
            reasoning="No text available for LLM scan.",
            assessment="Cannot assess — no clinical text.",
        )

    try:
        response = llm.invoke(MED_TEXT_PROMPT.format(text=text))
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
            return MedicationAgentOutput(
                meds_found=found,
                medications=parsed.get("medications", []),
                status=status,
                source="text",
                confidence=conf,
                confidence_score=_compute_confidence_score(found, "text", status),
                reasoning=str(parsed.get("reasoning", "")),
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
        confidence_score=0.0,
        reasoning="LLM scan failed.",
        assessment="LLM scan failed — defaulting negative.",
    )
