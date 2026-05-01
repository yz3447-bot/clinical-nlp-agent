"""
eval/llm_judge.py — LLM-as-judge for AD/ADRD Synthesizer reasoning quality.

Evaluates cases that have agent discrepancies along three dimensions:
  1. reasoning_clarity        — can each decision be traced to specific evidence?
  2. contradiction_handling   — is the overrule justification sufficient?
  3. evidence_consistency     — does the conclusion match the cited evidence?
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger

logger = get_logger(__name__)

# ─── Judge prompt ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """你是一位临床AI系统评审专家。请评估以下AD/ADRD诊断系统的推理链质量。

预测结果：{prediction}（1=有AD，0=无AD）
实际标签：{ground_truth}
Agent矛盾情况：{discrepancy}
推理链：{causal_chain}
被overrule的agent：{overruled_agents}

请从以下三个维度打分（每项0-1）：
1. 推理逻辑是否清晰（能否追溯到具体证据）
2. 矛盾处理是否合理（overrule的理由是否充分）
3. 结论与证据是否一致

输出JSON：
{{
  "reasoning_clarity": float,
  "contradiction_handling": float,
  "evidence_consistency": float,
  "overall_score": float,
  "judge_comment": str
}}"""

_CAUSAL_CHAIN_CHAR_LIMIT = 2000  # avoid sending extremely long chains to the LLM


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json(content: str) -> dict | None:
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


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return default


# ─── Core judge functions ──────────────────────────────────────────────────────

def judge_case(row: dict, llm) -> dict:
    """Evaluate a single case with the LLM judge.

    Args:
        row: A dict with keys from predictions_c.csv
        llm: LangChain LLM instance

    Returns:
        Judge result dict with scores and comment.
    """
    note_id    = str(row.get("note_id", ""))
    gt         = row.get("ground_truth", "")
    pred       = row.get("final_prediction", "")
    is_correct = str(gt) == str(pred)

    prompt = JUDGE_PROMPT.format(
        prediction=pred,
        ground_truth=gt,
        discrepancy=row.get("discrepancy", "(none)") or "(none)",
        causal_chain=str(row.get("causal_chain", ""))[:_CAUSAL_CHAIN_CHAR_LIMIT],
        overruled_agents=row.get("overruled_agents", "[]") or "[]",
    )

    try:
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        parsed   = _parse_json(content)

        if parsed:
            return {
                "note_id":                 note_id,
                "ground_truth":            gt,
                "final_prediction":        pred,
                "is_correct":              is_correct,
                "reasoning_clarity":       _safe_float(parsed.get("reasoning_clarity")),
                "contradiction_handling":  _safe_float(parsed.get("contradiction_handling")),
                "evidence_consistency":    _safe_float(parsed.get("evidence_consistency")),
                "overall_score":           _safe_float(parsed.get("overall_score")),
                "judge_comment":           str(parsed.get("judge_comment", "")),
                "discrepancy":             str(row.get("discrepancy", "")),
            }
    except Exception as e:
        logger.warning("LLM judge call failed: %s", e)

    # Fallback: judge call failed
    return {
        "note_id":                 note_id,
        "ground_truth":            gt,
        "final_prediction":        pred,
        "is_correct":              is_correct,
        "reasoning_clarity":       0.0,
        "contradiction_handling":  0.0,
        "evidence_consistency":    0.0,
        "overall_score":           0.0,
        "judge_comment":           "[LLM judge call failed]",
        "discrepancy":             str(row.get("discrepancy", "")),
    }


def run_llm_judge(
    cases: list[dict],
    llm,
    delay_seconds: float = 5.0,
) -> list[dict]:
    """Run the LLM judge on a list of cases with rate-limit spacing.

    Args:
        cases:         List of row dicts from the predictions DataFrame.
        llm:           Initialised LangChain LLM instance.
        delay_seconds: Seconds to wait between LLM calls (rate-limit buffer).

    Returns:
        List of judge result dicts, one per input case.
    """
    results = []
    for i, case in enumerate(cases):
        note_id = case.get("note_id", f"#{i}")
        print(f"  [Judge {i+1}/{len(cases)}] {note_id}", end=" ... ", flush=True)
        result = judge_case(case, llm)
        results.append(result)
        print(f"score={result['overall_score']:.2f}")
        if i < len(cases) - 1:
            time.sleep(delay_seconds)

    return results
