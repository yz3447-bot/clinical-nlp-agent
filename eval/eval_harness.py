"""
eval/eval_harness.py — Evaluation harness for the AD/ADRD multi-agent system.

Reads an existing predictions CSV (does NOT re-run the pipeline) and produces:
  - Overall metrics (accuracy, sensitivity, PPV, specificity, F1)
  - Confusion matrix
  - Error analysis across five buckets
  - Optional LLM-as-judge scoring for contradictory cases
  - Self-contained HTML report in eval/reports/

Usage:
    # Basic eval (no API key needed)
    python eval/eval_harness.py --predictions outputs/predictions_c.csv

    # Full eval with LLM judge
    python eval/eval_harness.py --predictions outputs/predictions_c.csv --run-llm-judge
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from the project root or from eval/
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.report_generator import generate_report
from logger import get_logger

logger = get_logger(__name__)


# ─── Data loading ─────────────────────────────────────────────────────────────

_OPTIONAL_COLS = {
    "synthesis_mode":        "unknown",
    "confidence_score_clin": 0.0,
    "confidence_score_med":  0.0,
    "confidence_score_dx":   0.0,
    "confidence_correction": "",
}


def _safe_json_list(val) -> list:
    if not val or str(val).strip() in ("", "[]"):
        return []
    s = str(val).replace("'", '"')
    try:
        result = json.loads(s)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions CSV, normalise types, inject missing columns."""
    df = pd.read_csv(path, dtype=str).fillna("")

    # Keep only rows with definitive labels (exclude uncertain -1)
    df = df[df["ground_truth"].isin(["0", "1"])].copy()
    if df.empty:
        logger.warning("No labeled rows found (ground_truth must be 0 or 1).")
        sys.exit(0)

    df["ground_truth"]     = df["ground_truth"].astype(int)
    df["final_prediction"] = df["final_prediction"].astype(int)

    # Inject optional columns added in Stage 8 (may be absent in older CSVs)
    for col, default in _OPTIONAL_COLS.items():
        if col not in df.columns:
            df[col] = default

    for col in ("confidence_score_clin", "confidence_score_med", "confidence_score_dx"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    total   = len(df)
    correct = int((df["ground_truth"] == df["final_prediction"]).sum())

    tp = int(((df["final_prediction"] == 1) & (df["ground_truth"] == 1)).sum())
    fp = int(((df["final_prediction"] == 1) & (df["ground_truth"] == 0)).sum())
    fn = int(((df["final_prediction"] == 0) & (df["ground_truth"] == 1)).sum())
    tn = int(((df["final_prediction"] == 0) & (df["ground_truth"] == 0)).sum())

    accuracy    = correct / total          if total        > 0 else 0.0
    sensitivity = tp / (tp + fn)          if (tp + fn)    > 0 else 0.0
    ppv         = tp / (tp + fp)          if (tp + fp)    > 0 else 0.0
    specificity = tn / (tn + fp)          if (tn + fp)    > 0 else 0.0
    f1          = (2 * ppv * sensitivity / (ppv + sensitivity)
                   if (ppv + sensitivity) > 0 else 0.0)

    return {
        "total": total, "correct": correct,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy, "sensitivity": sensitivity,
        "ppv": ppv, "specificity": specificity, "f1": f1,
    }


# ─── Error bucket analysis ────────────────────────────────────────────────────

def build_error_buckets(df: pd.DataFrame) -> dict:
    errors = df[df["ground_truth"] != df["final_prediction"]].copy()

    has_discrepancy = df["discrepancy"].str.strip() != ""
    is_low_conf     = df["confidence"] == "low"
    is_consensus    = df["synthesis_mode"].isin(
        ["consensus_positive", "consensus_negative"]
    )

    return {
        # Prediction-error buckets
        "false_positives":       errors[errors["final_prediction"] == 1].to_dict("records"),
        "false_negatives":       errors[errors["final_prediction"] == 0].to_dict("records"),
        "contradictory_errors":  errors[has_discrepancy[errors.index]].to_dict("records"),
        "low_confidence_errors": errors[is_low_conf[errors.index]].to_dict("records"),
        "consensus_errors":      errors[is_consensus[errors.index]].to_dict("records"),
        # All contradictory cases (correct + incorrect) — used for LLM judge
        "all_contradictory":     df[has_discrepancy].to_dict("records"),
    }


# ─── Stdout summary ───────────────────────────────────────────────────────────

def _print_summary(metrics: dict, buckets: dict) -> None:
    m = metrics
    sep = "=" * 58
    logger.info(sep)
    logger.info("  AD/ADRD EVAL RESULTS")
    logger.info(sep)
    logger.info("  Total labeled      : %d", m["total"])
    logger.info("  Accuracy           : %.4f  (%d/%d)", m["accuracy"], m["correct"], m["total"])
    logger.info("  Sensitivity        : %.4f", m["sensitivity"])
    logger.info("  PPV (Precision)    : %.4f", m["ppv"])
    logger.info("  Specificity        : %.4f", m["specificity"])
    logger.info("  F1                 : %.4f", m["f1"])
    logger.info(sep)
    logger.info("  Confusion Matrix:")
    logger.info("    TP=%d  FN=%d", m["tp"], m["fn"])
    logger.info("    FP=%d  TN=%d", m["fp"], m["tn"])
    logger.info(sep)
    logger.info("  Error Buckets:")
    total_err = len(buckets["false_positives"]) + len(buckets["false_negatives"])
    logger.info("    Total errors         : %d", total_err)
    logger.info("    False Positives      : %d", len(buckets["false_positives"]))
    logger.info("    False Negatives      : %d", len(buckets["false_negatives"]))
    logger.info("    Contradictory errors : %d", len(buckets["contradictory_errors"]))
    logger.info("    Low-confidence errs  : %d", len(buckets["low_confidence_errors"]))
    logger.info("    Consensus path errs  : %d", len(buckets["consensus_errors"]))
    logger.info("    All w/ discrepancy   : %d", len(buckets["all_contradictory"]))
    logger.info(sep)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AD/ADRD Eval Harness — reads predictions CSV, no pipeline re-run."
    )
    parser.add_argument(
        "--input", default="data/data_test.csv",
        help="Input EHR CSV (unused by default — kept for future re-run mode).",
    )
    parser.add_argument(
        "--predictions", default="outputs/predictions_c.csv",
        help="Path to existing predictions CSV (default: outputs/predictions_c.csv).",
    )
    parser.add_argument(
        "--output", default="eval/reports/",
        help="Directory for the HTML report (default: eval/reports/).",
    )
    parser.add_argument(
        "--run-llm-judge", action="store_true", default=False,
        help="Enable LLM-as-judge scoring for contradictory cases (requires GEMINI_API_KEY).",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.warning("Predictions file not found: %s. Nothing to evaluate.", pred_path)
        sys.exit(0)

    # ── Load & compute ────────────────────────────────────────────────────────
    logger.info("Loading predictions: %s", pred_path)
    df      = load_predictions(pred_path)
    metrics = compute_metrics(df)
    buckets = build_error_buckets(df)

    _print_summary(metrics, buckets)

    # ── Optional LLM judge ────────────────────────────────────────────────────
    judge_results: list[dict] = []

    if args.run_llm_judge:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — skipping LLM judge.")
        elif not buckets["all_contradictory"]:
            logger.info("No contradictory cases found — nothing to judge.")
        else:
            logger.info(
                "Scoring %d contradictory cases with LLM judge...",
                len(buckets["all_contradictory"]),
            )
            from main_c import build_llm
            from eval.llm_judge import run_llm_judge

            llm = build_llm()
            judge_results = run_llm_judge(buckets["all_contradictory"], llm)

            avg = sum(r["overall_score"] for r in judge_results) / len(judge_results)
            logger.info("LLM judge average reasoning quality score: %.3f", avg)

    # ── Generate HTML report ──────────────────────────────────────────────────
    report_path = generate_report(
        metrics=metrics,
        buckets=buckets,
        judge_results=judge_results,
        df=df,
        output_dir=Path(args.output),
    )
    logger.info("Report saved: %s", report_path)

    # ── CI gate: sensitivity threshold ───────────────────────────────────────
    threshold = 0.90
    if metrics["sensitivity"] < threshold:
        logger.error(
            "[FAIL] Sensitivity %.4f is below threshold %.2f",
            metrics["sensitivity"], threshold,
        )
        sys.exit(1)

    logger.info("[PASS] Sensitivity %.4f >= %.2f threshold.", metrics["sensitivity"], threshold)


if __name__ == "__main__":
    main()
