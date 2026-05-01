"""
main_c.py — Horizontal Multi-Agent System for AD/ADRD Identification from EHR records.

Architecture (three independent specialist agents + synthesizer):
  Layer 1  ClinTextAgent   (neurologist,    weight +3)  ─┐
  Layer 2  MedicationAgent (pharmacist,     weight +3)  ─┤─→ SynthesizerAgent → prediction + subtype
  Layer 3  DiagnosisAgent  (coding spec,    weight +1)  ─┘
  Agents run in parallel via ThreadPoolExecutor (inside pipeline.py).
"""

import csv
import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

from logger import get_logger
from pipeline import run_pipeline, EVIDENCE_COLUMNS
from rag.knowledge_base import KnowledgeBase
from rag.seed_data import SEED_CASES
from schemas import PipelineResult

logger = get_logger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_CSV    = OUTPUT_DIR / "predictions_c.csv"
EVIDENCE_CSV  = OUTPUT_DIR / "agent_evidence.csv"
WAIT_BETWEEN_ROWS = 30  # seconds between records (rate limit buffer)

# Driven entirely by PipelineResult — no manual field list to maintain
OUTPUT_COLUMNS = list(PipelineResult.model_fields.keys())

# Ground truth column name fragments
_GT_ADRD_FRAG    = "is AD/ADRD"
_GT_SUBTYPE_FRAG = "is subtype"


def build_llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set. "
            "Export it before running: export GEMINI_API_KEY=your_key"
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )


def find_csv_files() -> list[Path]:
    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Place your data CSV there.")
    return csvs


def load_processed_note_ids() -> set:
    """Load already-processed note_ids for checkpoint/resume."""
    if not OUTPUT_CSV.exists():
        return set()
    processed = set()
    with open(OUTPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(str(row["note_id"]))
    logger.info("[Checkpoint] Resuming — %d records already processed.", len(processed))
    return processed


def init_output_csv():
    """Create output CSV with header if missing (checks file size, not just existence)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()


def init_evidence_csv():
    """Create agent_evidence.csv with header if missing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not EVIDENCE_CSV.exists() or EVIDENCE_CSV.stat().st_size == 0:
        with open(EVIDENCE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVIDENCE_COLUMNS)
            writer.writeheader()


def append_evidence(record: dict):
    """Append a single evidence row to agent_evidence.csv."""
    with open(EVIDENCE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVIDENCE_COLUMNS)
        writer.writerow(record)


def _serialise_for_csv(data: dict) -> dict:
    """Convert model_dump() output to CSV-safe types.

    - list  → JSON string  (so DictWriter writes '[\"a\",\"b\"]' not \"['a','b']\")
    - None  → ""           (avoids 'None' string in CSV)
    - bool  → int          (True/False → 1/0, backward-compatible with existing CSVs)
    """
    out = {}
    for key, val in data.items():
        if isinstance(val, list):
            out[key] = json.dumps(val)
        elif val is None:
            out[key] = ""
        elif isinstance(val, bool):        # must precede int check; bool ⊂ int
            out[key] = int(val)
        else:
            out[key] = val
    return out


def append_result(result: dict):
    """Append a single result row to the output CSV."""
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writerow(result)


def process_row(row: dict, llm, kb=None, run_id: str = "") -> dict:
    """
    Extract fields from a CSV row, run the pipeline, return a CSV-ready dict.
    All agent execution and orchestration logic lives in pipeline.run_pipeline().
    """
    note_id    = str(row.get("note_id", ""))
    subject_id = str(row.get("subject_id", ""))

    # Ground truth — AD/ADRD label
    gt_val = next(
        (row[col] for col in row if _GT_ADRD_FRAG in col),
        row.get("label", row.get("ground_truth", "")),
    )
    ground_truth = str(gt_val).strip()

    # Ground truth — subtype label
    gt_subtype_val = next(
        (row[col] for col in row if _GT_SUBTYPE_FRAG in col),
        "",
    )
    ground_truth_subtype = str(gt_subtype_val).strip()

    text      = str(row.get("text", "") or "")
    icd_codes = str(row.get("all_icd_codes", "") or "")
    med_annotation_col = next(
        (c for c in row_dict if "AD/ADRD medications" in c), None
    )
    ad_med_annotation = str(row_dict.get(med_annotation_col, "")) if med_annotation_col else ""

    logger.info(
        "[Processing] note_id=%s | subject_id=%s | gt=%s | gt_subtype=%s | text_len=%d",
        note_id, subject_id, ground_truth, ground_truth_subtype, len(text),
    )

    pipeline_result, _, evidence_record = run_pipeline(
        text=text,
        icd_codes=icd_codes,
        llm=llm,
        kb=kb,
        run_id=run_id,
        note_id=note_id,
        subject_id=subject_id,
        ground_truth=ground_truth,
        ground_truth_subtype=ground_truth_subtype,
        ad_med_annotation=ad_med_annotation,
    )
    append_evidence(evidence_record)
    return _serialise_for_csv(pipeline_result.model_dump())


def compute_metrics(output_csv: Path):
    """Log AD/ADRD diagnosis metrics (accuracy, sensitivity, PPV)."""
    df = pd.read_csv(output_csv, dtype=str)

    df_eval = df[df["ground_truth"].astype(str).isin(["0", "1"])].copy()
    df_eval["ground_truth"]     = df_eval["ground_truth"].astype(int)
    df_eval["final_prediction"] = df_eval["final_prediction"].astype(int)

    total = len(df_eval)
    if total == 0:
        logger.warning("No valid rows to compute metrics.")
        return

    correct     = (df_eval["ground_truth"] == df_eval["final_prediction"]).sum()
    accuracy    = correct / total
    tp = ((df_eval["final_prediction"] == 1) & (df_eval["ground_truth"] == 1)).sum()
    fp = ((df_eval["final_prediction"] == 1) & (df_eval["ground_truth"] == 0)).sum()
    fn = ((df_eval["final_prediction"] == 0) & (df_eval["ground_truth"] == 1)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    logger.info(
        "\n" + "=" * 55 + "\n"
        "  AD/ADRD DIAGNOSIS METRICS\n"
        + "=" * 55 + "\n"
        + f"  Total evaluated  : {total}\n"
        + f"  Accuracy         : {accuracy:.4f}  ({correct}/{total})\n"
        + f"  Sensitivity      : {sensitivity:.4f}\n"
        + f"  PPV (Precision)  : {ppv:.4f}\n"
        + "=" * 55
    )


def main():
    logger.info("=" * 60)
    logger.info("  AD/ADRD Horizontal Multi-Agent System  (main_c.py)")
    logger.info("=" * 60)

    llm = build_llm()
    logger.info("[LLM] Gemini-2.5-flash initialized.")

    _kb = KnowledgeBase()
    for case in SEED_CASES:
        _kb.add_case(
            note_id=case["note_id"],
            text=case["text"],
            label=case["label"],
            subtype=case["subtype"],
        )
    logger.info("[RAG] KnowledgeBase initialized with %d seed cases.", _kb.get_stats())

    csv_files = find_csv_files()
    logger.info("[Data] Found %d CSV file(s): %s", len(csv_files), [f.name for f in csv_files])

    init_output_csv()
    init_evidence_csv()
    processed_ids = load_processed_note_ids()

    # One run_id shared across all records in this invocation
    run_id = str(uuid.uuid4())
    logger.info("[Run] run_id=%s", run_id)

    total_processed     = 0
    total_skipped_label = 0
    total_skipped_done  = 0

    for csv_path in csv_files:
        logger.info("[Data] Loading: %s", csv_path.name)
        df = pd.read_csv(csv_path, dtype=str)
        df = df.fillna("")

        gt_col = next((c for c in df.columns if _GT_ADRD_FRAG in c), None)
        if gt_col is None:
            gt_col = next((c for c in df.columns if c.lower() in ("label", "ground_truth")), None)
        logger.info("[Data] Ground truth column: '%s'", gt_col)

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            note_id  = str(row_dict.get("note_id", idx))

            # Skip uncertain labels
            if gt_col:
                label_val = str(row_dict.get(gt_col, "")).strip()
                if label_val == "-1":
                    total_skipped_label += 1
                    continue

            # Skip already processed
            if note_id in processed_ids:
                total_skipped_done += 1
                continue

            try:
                result = process_row(row_dict, llm, kb=_kb, run_id=run_id)
                append_result(result)
                processed_ids.add(note_id)
                total_processed += 1

                logger.info(
                    "  -> Saved result for note_id=%s. Waiting %ds...",
                    note_id, WAIT_BETWEEN_ROWS,
                )
                time.sleep(WAIT_BETWEEN_ROWS)

            except Exception as e:
                logger.error("[ERROR] note_id=%s: %s", note_id, e)
                continue

    logger.info(
        "[Done] Processed: %d | Skipped (label=-1): %d | Skipped (already done): %d",
        total_processed, total_skipped_label, total_skipped_done,
    )
    logger.info("[Output] Results saved to: %s", OUTPUT_CSV)

    compute_metrics(OUTPUT_CSV)


if __name__ == "__main__":
    main()
