"""
main_c.py — Horizontal Multi-Agent System for AD/ADRD Identification from EHR records.

Architecture (three independent specialist agents + synthesizer):
  Layer 1  ClinTextAgent   (neurologist,    weight +3)  ─┐
  Layer 2  MedicationAgent (pharmacist,     weight +3)  ─┤─→ SynthesizerAgent → prediction + subtype
  Layer 3  DiagnosisAgent  (coding spec,    weight +1)  ─┘
  Agents run in parallel via ThreadPoolExecutor.
"""

import os
import csv
import json
import time
import concurrent.futures
from pathlib import Path

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.diagnosis_agent import run_diagnosis_agent
from agents.medication_agent import run_medication_agent
from agents.clintext_agent import run_clintext_agent
from agents.synthesizer_agent import run_synthesizer_agent

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "predictions_c.csv"
WAIT_BETWEEN_ROWS = 30  # seconds between records (rate limit buffer)

OUTPUT_COLUMNS = [
    "note_id", "subject_id", "ground_truth", "ground_truth_subtype",
    "dx_found", "meds_found", "symptoms_found",
    "final_prediction", "subtype", "confidence",
    "contributing_agents", "causal_chain", "discrepancy",
    "overruled_agents", "reason",
]

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
    print(f"[Checkpoint] Resuming — {len(processed)} records already processed.")
    return processed


def init_output_csv():
    """Create output CSV with header if missing (checks file size, not just existence)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()


def append_result(result: dict):
    """Append a single result row to the output CSV."""
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writerow(result)


def process_row(row: dict, llm) -> dict:
    """
    Run all 3 specialist agents in parallel, then the Synthesizer serially.
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

    text = str(row.get("text", "") or "")

    print(f"[Processing] note_id={note_id} | subject_id={subject_id} "
          f"| gt={ground_truth} | gt_subtype={ground_truth_subtype} "
          f"| text_len={len(text)}")

    row_dict = dict(row)

    # ── Parallel execution: Agents 1–3 ──────────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_dx   = executor.submit(run_diagnosis_agent,  row_dict, llm)
        future_med  = executor.submit(run_medication_agent, row_dict, llm)
        future_clin = executor.submit(run_clintext_agent,   row_dict, llm)

        dx_result   = future_dx.result()
        med_result  = future_med.result()
        clin_result = future_clin.result()

    print(f"  [L3-DiagnosisAgent]  dx_found={dx_result['dx_found']} "
          f"current={dx_result.get('is_current_diagnosis','?')} "
          f"conf={dx_result['confidence']} codes={dx_result['matched_codes']}")
    print(f"  [L2-MedicationAgent] meds_found={med_result['meds_found']} "
          f"status={med_result.get('status','?')} "
          f"meds={med_result['medications']} source={med_result.get('source','?')}")
    ev_count = len(clin_result.get("evidence_list", []))
    print(f"  [L1-ClinTextAgent]   symptoms_found={clin_result['symptoms_found']} "
          f"conf={clin_result['confidence']} evidence_count={ev_count}")

    # ── Serial: Synthesizer ───────────────────────────────────────────────────
    synth_result = run_synthesizer_agent(dx_result, med_result, clin_result, llm, text=text)

    print(f"  [Synthesizer]  prediction={synth_result['final_prediction']} "
          f"subtype={synth_result['subtype']} "
          f"confidence={synth_result['confidence']}")
    print(f"  [Causal Chain] {synth_result['causal_chain'][:200]}...")
    if synth_result.get("discrepancy"):
        print(f"  [Discrepancy]  {synth_result['discrepancy']}")
    if synth_result.get("overruled_agents"):
        print(f"  [Overruled]    {synth_result['overruled_agents']}")

    return {
        "note_id":              note_id,
        "subject_id":           subject_id,
        "ground_truth":         ground_truth,
        "ground_truth_subtype": ground_truth_subtype,
        "dx_found":             int(dx_result["dx_found"]),
        "meds_found":           int(med_result["meds_found"]),
        "symptoms_found":       int(clin_result["symptoms_found"]),
        "final_prediction":     synth_result["final_prediction"],
        "subtype":              synth_result["subtype"],
        "confidence":           synth_result["confidence"],
        "contributing_agents":  json.dumps(synth_result["contributing_agents"]),
        "causal_chain":         synth_result["causal_chain"],
        "discrepancy":          synth_result.get("discrepancy") or "",
        "overruled_agents":     json.dumps(synth_result.get("overruled_agents", [])),
        "reason":               synth_result["reason"],
    }


def compute_metrics(output_csv: Path):
    """Print AD/ADRD diagnosis metrics (accuracy, sensitivity, PPV)."""
    df = pd.read_csv(output_csv, dtype=str)

    df_eval = df[df["ground_truth"].astype(str).isin(["0", "1"])].copy()
    df_eval["ground_truth"]     = df_eval["ground_truth"].astype(int)
    df_eval["final_prediction"] = df_eval["final_prediction"].astype(int)

    total = len(df_eval)
    if total == 0:
        print("No valid rows to compute metrics.")
        return

    correct     = (df_eval["ground_truth"] == df_eval["final_prediction"]).sum()
    accuracy    = correct / total
    tp = ((df_eval["final_prediction"] == 1) & (df_eval["ground_truth"] == 1)).sum()
    fp = ((df_eval["final_prediction"] == 1) & (df_eval["ground_truth"] == 0)).sum()
    fn = ((df_eval["final_prediction"] == 0) & (df_eval["ground_truth"] == 1)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    print("\n" + "=" * 55)
    print("  AD/ADRD DIAGNOSIS METRICS")
    print("=" * 55)
    print(f"  Total evaluated  : {total}")
    print(f"  Accuracy         : {accuracy:.4f}  ({correct}/{total})")
    print(f"  Sensitivity      : {sensitivity:.4f}")
    print(f"  PPV (Precision)  : {ppv:.4f}")
    print("=" * 55 + "\n")


def main():
    print("=" * 60)
    print("  AD/ADRD Horizontal Multi-Agent System  (main_c.py)")
    print("=" * 60)

    llm = build_llm()
    print("[LLM] Gemini-2.5-flash initialized.")

    csv_files = find_csv_files()
    print(f"[Data] Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")

    init_output_csv()
    processed_ids = load_processed_note_ids()

    total_processed    = 0
    total_skipped_label = 0
    total_skipped_done  = 0

    for csv_path in csv_files:
        print(f"\n[Data] Loading: {csv_path.name}")
        df = pd.read_csv(csv_path, dtype=str)
        df = df.fillna("")

        gt_col = next((c for c in df.columns if _GT_ADRD_FRAG in c), None)
        if gt_col is None:
            gt_col = next((c for c in df.columns if c.lower() in ("label", "ground_truth")), None)
        print(f"[Data] Ground truth column: '{gt_col}'")

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
                result = process_row(row_dict, llm)
                append_result(result)
                processed_ids.add(note_id)
                total_processed += 1

                print(f"  -> Saved result for note_id={note_id}. "
                      f"Waiting {WAIT_BETWEEN_ROWS}s...")
                time.sleep(WAIT_BETWEEN_ROWS)

            except Exception as e:
                print(f"[ERROR] note_id={note_id}: {e}")
                continue

    print(f"\n[Done] Processed: {total_processed} | "
          f"Skipped (label=-1): {total_skipped_label} | "
          f"Skipped (already done): {total_skipped_done}")
    print(f"[Output] Results saved to: {OUTPUT_CSV}")

    compute_metrics(OUTPUT_CSV)


if __name__ == "__main__":
    main()
