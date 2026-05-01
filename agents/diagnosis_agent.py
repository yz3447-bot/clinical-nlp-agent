"""
Diagnosis Agent — Layer 3 (corroborative, weight +1).
Step 1: Rule-based ICD code matching (no LLM, cheap).
Step 2: LLM reads full note (including PMH) to determine whether codes
        represent genuine chronic AD/ADRD vs an exclusion case.
"""

import json
import re

from logger import get_logger
from schemas import DiagnosisAgentOutput

logger = get_logger(__name__)


def _extract_dx_sections(text: str) -> str:
    """
    Extract diagnosis-related sections from a clinical note.
    Targets: Assessment & Plan, Brief Hospital Course (MIMIC primary A&P),
             Active Issues, Discharge Diagnosis, Discharge Condition, Final Diagnosis.
    Falls back to the first 2000 chars when no section is found.
    Brief Hospital Course is added because MIMIC discharge summaries use it
    as the primary narrative A&P equivalent; 'Assessment & Plan' rarely appears.
    Active Issues covers MIMIC notes that use ACTIVE ISSUES: / INACTIVE ISSUES: headers.
    """
    patterns = [
        r"(?i)(past\s+medical\s+history.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
        r"(?i)(assessment\s*(?:and|&)\s*plan.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
        r"(?i)(brief\s+hospital\s+course.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
        r"(?i)(active\s+issues?.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
        r"(?i)(discharge\s+diagnosis.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
        r"(?i)(discharge\s+condition.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
        r"(?i)(final\s+diagnosis.*?)(?=\n[A-Z][A-Za-z\s&/]{2,}:\s*\n|\Z)",
    ]

    extracted = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        extracted.extend(m.strip() for m in matches if m.strip())

    if extracted:
        combined = "\n\n".join(dict.fromkeys(extracted))
        return combined[:4000]

    return text[:2000]


# ICD-9 AD/ADRD codes
AD_ICD9 = {
    "2900", "29010", "29011", "29020", "29021", "2903",
    "29040", "29041", "29042", "29043",
    "2940", "29410", "29411", "29420", "29421",
    "3310", "3311", "3312", "3316", "33182", "3349", "34830",
    "2930", "2931",
}

# ICD-10 AD/ADRD codes
AD_ICD10 = {
    "F0280", "F0281", "F0390", "F0391",
    "G309",
    "G3000", "G3001", "G3009",
    "G3010", "G3011", "G3019",
    "G308",
    "G3183", "G3189", "G319",
    "F03", "F030", "F039",
    "F0150", "F0151",
    "G30", "G31", "F02",
}

AD_ICD_WHITELIST = AD_ICD9 | AD_ICD10

DX_CONTEXT_PROMPT = """You are a medical coding specialist and diagnostician.

You have been provided with:
1. The patient's ICD codes that matched the AD/ADRD whitelist: {matched_codes}
2. Relevant sections of the clinical note (including Past Medical History)

Your task is to determine whether these ICD codes represent genuine CHRONIC AD/ADRD
in this patient, drawing on ALL documentation in the note.

As an experienced coding specialist, you know that:
- A chronic AD/ADRD diagnosis documented ANYWHERE in the note IS valid evidence.
  This includes Past Medical History, prior admission carry-forward codes for a
  genuine neurodegenerative condition, or any note section referencing dementia.
- ICD codes carried forward from prior admissions still count when they reflect a
  real chronic condition the patient has been living with.
- The fact that this admission is NOT primarily for AD/ADRD does NOT mean the
  patient lacks chronic AD/ADRD — patients with dementia are frequently admitted
  for other reasons (falls, infection, cardiac events).
- Normal mental status at discharge does NOT negate a pre-existing chronic
  dementia diagnosis — patients may return to their baseline.
- The following are EXCLUSION CASES — set is_chronic=false and is_exclusion=true:
  * Acute delirium (F05) or acute toxic/metabolic encephalopathy (G92) with NO
    separate documentation of underlying chronic dementia anywhere in the note
  * Record where the patient died acutely and the discharge diagnosis contains no
    AD/ADRD, and the note has no cognitive content whatsoever
  * G312 (alcohol-related neurodegeneration) where the note describes only acute
    alcohol-related events with no chronic dementia description

Carefully read ALL sections, especially:
- Past Medical History — prior chronic diagnoses are valid evidence
- Assessment & Plan and Discharge Diagnosis
- Active problem list and Brief Hospital Course
- History of Present Illness

Clinical note sections:
---
{text}
---

Output ONLY this JSON:
{{
  "dx_found": true,
  "matched_codes": {matched_codes_json},
  "is_chronic": true or false,
  "is_exclusion": true or false,
  "confidence": "high" or "medium" or "low",
  "reasoning": "cite specific text explaining whether chronic AD/ADRD is supported, absent, or excluded",
  "assessment": "one sentence: coding evidence for or against chronic AD/ADRD"
}}

Confidence reflects your certainty in your own conclusion (is_chronic):
- high: clear documentation strongly supports or clearly refutes the chronic diagnosis
- medium: some evidence but with gaps or ambiguity
- low: minimal evidence; uncertain judgment

Set is_exclusion=true ONLY when is_chronic=false AND a specific exclusion condition
applies (acute delirium only, death carry-forward, or alcohol-only G312).
When is_chronic=true, always set is_exclusion=false."""


def _normalise(code: str) -> str:
    return code.replace(".", "").strip().upper()


def _compute_confidence_score(
    dx_found: bool,
    is_chronic: bool,
    confidence: str,
    is_exclusion: bool = False,
) -> float:
    """Confidence score = agent's certainty in its own conclusion (not signal strength).

    no ICD match          → 0.85  (rule-based, certain negative)
    is_chronic=False
      is_exclusion=True   → 0.85  (LLM identified a known exclusion pattern)
      is_exclusion=False  → 0.30  (codes present but LLM uncertain about chronicity)
    is_chronic=True
      high LLM certainty  → 0.90
      medium              → 0.70
      low                 → 0.40
    """
    if not dx_found:
        return 0.85
    if not is_chronic:
        return 0.85 if is_exclusion else 0.30
    return {"high": 0.90, "medium": 0.70, "low": 0.40}.get(confidence, 0.40)


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


def run_diagnosis_agent(row: dict, llm=None) -> DiagnosisAgentOutput:
    """
    Args:
        row: dict with 'all_icd_codes' (pipe-separated) and 'text' key
        llm: LangChain LLM instance (enables current vs historical analysis)
    Returns:
        DiagnosisAgentOutput validated instance
    """
    # ── Step 1: Rule-based ICD matching ──────────────────────────────────────
    raw = str(row.get("all_icd_codes", "") or "")
    codes = [_normalise(c) for c in raw.split("|") if c.strip()]

    matched = []
    for code in codes:
        for wl in AD_ICD_WHITELIST:
            if code == wl or code.startswith(wl):
                matched.append(code)
                break
    matched = list(dict.fromkeys(matched))
    dx_found = len(matched) > 0

    if not dx_found:
        return DiagnosisAgentOutput(
            dx_found=False,
            matched_codes=[],
            is_current_diagnosis=False,
            confidence="high",
            confidence_score=_compute_confidence_score(False, False, "high"),
            reasoning="No AD/ADRD ICD codes found in record.",
            assessment="No AD/ADRD ICD codes present.",
        )

    # ── Step 2: LLM context analysis (chronic vs exclusion) ──────────────────
    text = str(row.get("text", "") or "")
    if not text.strip() or llm is None:
        return DiagnosisAgentOutput(
            dx_found=True,
            matched_codes=matched,
            is_current_diagnosis=True,
            confidence="medium",
            confidence_score=_compute_confidence_score(True, True, "medium"),
            reasoning="ICD codes matched; no LLM context check available.",
            assessment=f"ICD codes matched ({', '.join(matched)}); assumed chronic.",
        )

    try:
        dx_text = _extract_dx_sections(text)
        prompt = DX_CONTEXT_PROMPT.format(
            matched_codes=", ".join(matched),
            matched_codes_json=json.dumps(matched),
            text=dx_text,
        )
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        parsed   = _parse_llm_json(content)

        if parsed:
            is_chronic    = bool(parsed.get("is_chronic", True))
            is_exclusion  = bool(parsed.get("is_exclusion", False))
            conf_raw      = str(parsed.get("confidence", "medium")).lower().strip()
            conf          = conf_raw if conf_raw in ("high", "medium", "low") else "medium"
            return DiagnosisAgentOutput(
                dx_found=True,
                matched_codes=matched,
                is_current_diagnosis=is_chronic,
                confidence=conf,
                confidence_score=_compute_confidence_score(True, is_chronic, conf, is_exclusion),
                reasoning=str(parsed.get("reasoning", "")),
                assessment=str(parsed.get("assessment", "")),
            )
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")

    # Fallback: codes found but LLM failed
    return DiagnosisAgentOutput(
        dx_found=True,
        matched_codes=matched,
        is_current_diagnosis=True,
        confidence="medium",
        confidence_score=_compute_confidence_score(True, True, "medium"),
        reasoning="ICD codes matched; LLM context check failed.",
        assessment=f"ICD codes matched ({', '.join(matched)}); LLM check failed.",
    )
