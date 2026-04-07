"""
Clinical Text Agent — Layer 1 (primary evidence, weight +3).
Neurologist reads the FULL clinical note for current AD/ADRD evidence.
Returns exact quotes, explicit reasoning, and exclusion list.
"""

import json
import re

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


def run_clintext_agent(row: dict, llm) -> dict:
    """
    Args:
        row: dict with 'text' key (full clinical note)
        llm: LangChain LLM instance
    Returns:
        {
          "symptoms_found": bool,
          "evidence_list": list[str],
          "reasoning": str,
          "assessment": str,
          "confidence": "high" | "medium" | "low",
          "weight": 3
        }
    """
    text = str(row.get("text", "") or "")

    if not text.strip():
        return {
            "symptoms_found": False,
            "evidence_list":  [],
            "reasoning":      "No clinical text provided.",
            "assessment":     "No clinical text available.",
            "confidence":     "low",
            "weight":         3,
        }

    try:
        response = llm.invoke(CLINTEXT_PROMPT.format(text=text))
        content  = response.content if hasattr(response, "content") else str(response)
        parsed   = _parse_llm_json(content)

        if not parsed:
            raise ValueError("Unparseable LLM response")

        found      = bool(parsed.get("symptoms_found", False))
        conf       = parsed.get("confidence", "low")
        evidence   = parsed.get("evidence_list", [])
        reasoning  = str(parsed.get("reasoning", ""))
        assessment = str(parsed.get("assessment", ""))

        if found and not evidence:
            conf = "low"

        return {
            "symptoms_found": found,
            "evidence_list":  evidence,
            "reasoning":      reasoning,
            "assessment":     assessment,
            "confidence":     conf,
            "weight":         3,
        }

    except Exception as e:
        return {
            "symptoms_found": False,
            "evidence_list":  [],
            "reasoning":      f"Agent error: {e}",
            "assessment":     "Agent error — defaulting negative.",
            "confidence":     "low",
            "weight":         3,
        }
