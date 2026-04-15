"""
test_run.py — Quick smoke test: runs the full pipeline on the first 5 records.

Usage:
    python test_run.py                  # first 5 records, verbose output
    python test_run.py --n 3            # first 3 records
    python test_run.py --no-wait        # skip the 30s rate-limit buffer

No API key? Set it first:
    export GEMINI_API_KEY=your_key_here   # Mac/Linux
    set GEMINI_API_KEY=your_key_here      # Windows
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from main_c import build_llm, process_row, find_csv_files

SEPARATOR = "=" * 60


def print_result(i: int, result: dict):
    pred  = result["final_prediction"]
    label = "AD/ADRD POSITIVE ✓" if pred == 1 else "NEGATIVE ✗"
    print(f"\n{'─' * 60}")
    print(f"  Record {i+1}  |  note_id={result['note_id']}  |  {label}")
    print(f"  Subtype     : {result['subtype']}")
    print(f"  Confidence  : {result['confidence']}")
    print(f"  Agents      : dx={result['dx_found']}  "
          f"med={result['meds_found']}  "
          f"clin={result['symptoms_found']}")
    contrib = json.loads(result["contributing_agents"]) if result["contributing_agents"] else []
    print(f"  Contributing: {contrib}")
    if result.get("discrepancy"):
        print(f"  Discrepancy : {result['discrepancy']}")
    if result.get("overruled_agents"):
        overruled = json.loads(result["overruled_agents"])
        if overruled:
            print(f"  Overruled   : {overruled}")
    print(f"  Causal chain: {result['causal_chain'][:300]}...")
    if result.get("ground_truth") not in ("", None):
        match = "✓ correct" if str(result["ground_truth"]) == str(pred) else "✗ wrong"
        print(f"  Ground truth: {result['ground_truth']}  →  {match}")


def main():
    parser = argparse.ArgumentParser(description="Quick smoke test for the AD/ADRD agent pipeline.")
    parser.add_argument("--n",       type=int,  default=5,     help="Number of records to run (default: 5)")
    parser.add_argument("--no-wait", action="store_true",      help="Skip 30s rate-limit wait between records")
    args = parser.parse_args()

    print(SEPARATOR)
    print("  AD/ADRD Multi-Agent — Quick Test Run")
    print(SEPARATOR)

    llm = build_llm()
    print(f"[LLM] Gemini-2.5-flash initialized.\n")

    csv_files = find_csv_files()
    df = pd.read_csv(csv_files[0], dtype=str).fillna("")

    # Skip uncertain labels (-1)
    gt_col = next((c for c in df.columns if "is AD/ADRD" in c), None)
    if gt_col:
        df = df[df[gt_col].str.strip() != "-1"]

    n = min(args.n, len(df))
    print(f"[Data] {csv_files[0].name}  —  running first {n} records (of {len(df)} labeled)\n")

    results = []
    for i, (_, row) in enumerate(df.head(n).iterrows()):
        row_dict = row.to_dict()
        note_id  = row_dict.get("note_id", str(i))
        print(f"[{i+1}/{n}] Processing note_id={note_id} ...")

        try:
            result = process_row(row_dict, llm)
            results.append(result)
            print_result(i, result)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        if i < n - 1 and not args.no_wait:
            print(f"\n  Waiting 30s (rate-limit buffer)...")
            time.sleep(30)

    # Summary
    if results:
        print(f"\n{SEPARATOR}")
        print(f"  SUMMARY  ({len(results)}/{n} completed)")
        print(SEPARATOR)
        correct = sum(
            1 for r in results
            if r.get("ground_truth") not in ("", None)
            and str(r["ground_truth"]) == str(r["final_prediction"])
        )
        labeled = sum(1 for r in results if r.get("ground_truth") not in ("", None))
        if labeled:
            print(f"  Accuracy on {labeled} labeled records: "
                  f"{correct}/{labeled} = {correct/labeled:.1%}")
        pos = sum(1 for r in results if r["final_prediction"] == 1)
        print(f"  Predicted positive : {pos}/{len(results)}")
        print(f"  Predicted negative : {len(results)-pos}/{len(results)}")
        print(SEPARATOR)
    print("\n[Done] To run the full batch pipeline: python main_c.py\n")


if __name__ == "__main__":
    main()
