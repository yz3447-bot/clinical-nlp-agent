"""
Diagnosis Agent — Layer 3 (corroborative, weight +1).
Step 1: Rule-based ICD code matching (no LLM, cheap).
Step 2: LLM reads full note to determine current vs historical coding.
"""

import json
import re

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
2. The complete clinical note

Your task is to determine whether these ICD codes represent:
- A CURRENT ACTIVE diagnosis being managed during this admission
- A HISTORICAL diagnosis carried forward from previous encounters

As an experienced coding specialist, you know that:
- ICD codes in EHR are often carried forward from previous admissions
- A current diagnosis should be reflected in the clinical note's
  Assessment & Plan or Discharge Diagnosis
- Historical codes may appear in Past Medical History without
  active management
- The presence of an ICD code alone does not confirm active disease

Carefully read the clinical note, especially:
- Assessment & Plan section
- Discharge diagnosis
- Active problem list
- Whether the admission is related to cognitive/behavioral issues

Clinical note:
---
{text}
---

Output ONLY this JSON:
{{
  "dx_found": true,
  "matched_codes": {matched_codes_json},
  "is_current_diagnosis": true or false,
  "confidence": "high" or "medium" or "low",
  "reasoning": "explain whether this is current or historical based on note",
  "assessment": "one sentence: coding evidence for or against current AD/ADRD"
}}"""


def _normalise(code: str) -> str:
    return code.replace(".", "").strip().upper()


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


def run_diagnosis_agent(row: dict, llm=None) -> dict:
    """
    Args:
        row: dict with 'all_icd_codes' (pipe-separated) and 'text' key
        llm: LangChain LLM instance (enables current vs historical analysis)
    Returns:
        {
          "dx_found": bool,
          "matched_codes": list[str],
          "is_current_diagnosis": bool,
          "confidence": "high" | "medium" | "low",
          "reasoning": str,
          "assessment": str,
          "weight": 1
        }
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
        return {
            "dx_found":             False,
            "matched_codes":        [],
            "is_current_diagnosis": False,
            "confidence":           "low",
            "reasoning":            "No AD/ADRD ICD codes found in record.",
            "assessment":           "No AD/ADRD ICD codes present.",
            "weight":               1,
        }

    # ── Step 2: LLM context analysis (current vs historical) ─────────────────
    text = str(row.get("text", "") or "")
    if not text.strip() or llm is None:
        return {
            "dx_found":             True,
            "matched_codes":        matched,
            "is_current_diagnosis": True,
            "confidence":           "medium",
            "reasoning":            "ICD codes matched; no LLM context check available.",
            "assessment":           f"ICD codes matched ({', '.join(matched)}); assumed current.",
            "weight":               1,
        }

    try:
        prompt = DX_CONTEXT_PROMPT.format(
            matched_codes=", ".join(matched),
            matched_codes_json=json.dumps(matched),
            text=text,
        )
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        parsed   = _parse_llm_json(content)

        if parsed:
            return {
                "dx_found":             True,
                "matched_codes":        matched,
                "is_current_diagnosis": bool(parsed.get("is_current_diagnosis", True)),
                "confidence":           parsed.get("confidence", "medium"),
                "reasoning":            str(parsed.get("reasoning", "")),
                "assessment":           str(parsed.get("assessment", "")),
                "weight":               1,
            }
    except Exception as e:
        pass

    # Fallback: codes found but LLM failed
    return {
        "dx_found":             True,
        "matched_codes":        matched,
        "is_current_diagnosis": True,
        "confidence":           "medium",
        "reasoning":            "ICD codes matched; LLM context check failed.",
        "assessment":           f"ICD codes matched ({', '.join(matched)}); LLM check failed.",
        "weight":               1,
    }
