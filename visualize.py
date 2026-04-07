"""
visualize.py v2 — Interactive report for AD/ADRD Horizontal Multi-Agent System.

Generates outputs/report.html with:
  Part 1: Interactive decision cards (badge popups, filter bar, search, load-more)
  Part 2: Performance dashboard (confusion matrix, grouped bar chart,
          disagreement analysis, optional system comparison)
"""

import json
import math
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"
REPORT_PATH = OUTPUT_DIR / "report.html"

# Agent metadata — mirrored into JS at runtime
AGENT_DEFS = [
    {
        "key": "symptoms_found",
        "name": "ClinTextAgent",
        "label": "L1 · Clinical Text",
        "agent_type": "LLM-based (Gemini 2.5 Flash)",
        "score": 3,
        "method": (
            "Gemini 2.5 Flash scans the full clinical note for explicit "
            "AD/ADRD clinical evidence: memory loss, disorientation, family-reported "
            "decline, functional impairment, behavioural change, language difficulty."
        ),
        "found_msg": (
            "Explicit AD/ADRD symptoms identified in the clinical text. "
            "High confidence = ≥2 distinct evidence categories. Score contribution: +3."
        ),
        "not_found_msg": (
            "No explicit AD/ADRD symptoms found in the note excerpt. Score contribution: +0."
        ),
    },
    {
        "key": "meds_found",
        "name": "MedicationAgent",
        "label": "L2 · Medications",
        "agent_type": "Hybrid (structured + LLM text)",
        "score": 3,
        "method": (
            "Phase 1: checks structured medication flag columns (donepezil, memantine, "
            "tacrine, rivastigmine, galantamine). Phase 2: if negative, Gemini scans "
            "the full clinical note for AD drug mentions including brand names."
        ),
        "found_msg": (
            "AD-specific medication prescribed — confirms clinical management of AD. "
            "Strong signal: clinician has already made the diagnosis. Score contribution: +3."
        ),
        "not_found_msg": (
            "No AD medications found in structured columns or clinical text. Score contribution: +0."
        ),
    },
    {
        "key": "dx_found",
        "name": "DiagnosisAgent",
        "label": "L3 · Diagnosis ICD",
        "agent_type": "Rule-based",
        "score": 1,
        "method": (
            "Scans the all_icd_codes field against the AD/ADRD ICD whitelist: "
            "G30, G300, G301, G308, G309, F02, F020, F0280, F0281, "
            "G31, G310, G311, G318. Prefix-match is used (e.g. G309 matches G30)."
        ),
        "found_msg": (
            "AD/ADRD-related ICD codes were matched in the diagnosis record. "
            "These codes are clinically assigned after diagnostic workup — "
            "a high-specificity signal. Score contribution: +3."
        ),
        "not_found_msg": (
            "No AD/ADRD ICD codes found. The patient may lack a formal diagnosis, "
            "codes may be in an untested variant format, or the condition is "
            "described narratively rather than coded. Score contribution: +0."
        ),
    },
]


# ── Data Processing ────────────────────────────────────────────────────────────

def load_predictions(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str)
    df = df[df["ground_truth"].astype(str).isin(["0", "1"])]
    if df.empty:
        return None
    df["ground_truth"] = df["ground_truth"].astype(int)
    df["final_prediction"] = df["final_prediction"].astype(int)
    for col in ["dx_found", "meds_found", "symptoms_found", "total_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {"accuracy": "N/A", "sensitivity": "N/A", "ppv": "N/A",
                "auc": "N/A", "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    correct = int((df["ground_truth"] == df["final_prediction"]).sum())
    tp = int(((df["final_prediction"] == 1) & (df["ground_truth"] == 1)).sum())
    fp = int(((df["final_prediction"] == 1) & (df["ground_truth"] == 0)).sum())
    fn = int(((df["final_prediction"] == 0) & (df["ground_truth"] == 1)).sum())
    tn = int(((df["final_prediction"] == 0) & (df["ground_truth"] == 0)).sum())

    def safe(num, den):
        return num / den if den > 0 else float("nan")

    accuracy    = safe(correct, total)
    sensitivity = safe(tp, tp + fn)
    ppv         = safe(tp, tp + fp)
    specificity = safe(tn, tn + fp)
    auc         = safe(sensitivity + specificity, 2) if not (math.isnan(sensitivity) or math.isnan(specificity)) else float("nan")

    def fmt(v):
        return f"{v:.4f}" if not (isinstance(v, float) and math.isnan(v)) else "N/A"

    tp_pct = round(tp / total * 100, 1) if total else 0
    fp_pct = round(fp / total * 100, 1) if total else 0
    fn_pct = round(fn / total * 100, 1) if total else 0
    tn_pct = round(tn / total * 100, 1) if total else 0

    return {
        "accuracy": fmt(accuracy), "sensitivity": fmt(sensitivity),
        "ppv": fmt(ppv), "auc": fmt(auc),
        "total": total, "correct": correct,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tp_pct": tp_pct, "fp_pct": fp_pct, "fn_pct": fn_pct, "tn_pct": tn_pct,
    }


def compute_agent_contrib(df: pd.DataFrame) -> list:
    """Returns data for the grouped bar chart (flag rate + precision per agent)."""
    result = []
    total = len(df)
    for ag in AGENT_DEFS:
        col = ag["key"]
        flagged = df[col].sum()
        flag_rate = round(flagged / total * 100, 1) if total > 0 else 0
        flagged_rows = df[df[col] == 1]
        if len(flagged_rows) > 0:
            precision = round(
                (flagged_rows["ground_truth"] == 1).sum() / len(flagged_rows) * 100, 1
            )
        else:
            precision = 0
        result.append({
            "name": ag["name"],
            "label": ag["label"],
            "flag_rate": flag_rate,
            "precision": precision,
            "flagged_count": int(flagged),
        })
    return result


def compute_disagreement_data(df: pd.DataFrame) -> dict:
    """Rule agents (dx, meds) vs LLM agents (clin) signal disagreement."""
    df2 = df.copy()
    df2["rule_pos"] = ((df2["dx_found"] == 1) | (df2["meds_found"] == 1)).astype(int)
    df2["llm_pos"]  = (df2["symptoms_found"] == 1).astype(int)
    df2["disagree"] = (df2["rule_pos"] != df2["llm_pos"]).astype(int)

    disagree_rows = df2[df2["disagree"] == 1]
    total_disagree = len(disagree_rows)
    accuracy_disagree = 0.0
    if total_disagree > 0:
        accuracy_disagree = round(
            (disagree_rows["ground_truth"] == disagree_rows["final_prediction"]).sum()
            / total_disagree * 100, 1
        )

    cases = []
    for _, row in disagree_rows.iterrows():
        cases.append({
            "note_id": str(row["note_id"]),
            "subject_id": str(row.get("subject_id", "")),
            "rule_pos": int(row["rule_pos"]),
            "llm_pos": int(row["llm_pos"]),
            "dx_found": int(row["dx_found"]),
            "meds_found": int(row["meds_found"]),
            "symptoms_found": int(row["symptoms_found"]),
            "final_prediction": int(row["final_prediction"]),
            "ground_truth": int(row["ground_truth"]),
            "is_correct": int(row["ground_truth"] == row["final_prediction"]),
            "confidence": str(row.get("confidence", "")),
        })

    return {
        "total": total_disagree,
        "accuracy": accuracy_disagree,
        "cases": cases,
    }


def serialize_rows(df: pd.DataFrame) -> list:
    """Serialize all rows to JSON-safe dicts for JS consumption."""
    rows = []
    for _, row in df.iterrows():
        contributing = []
        try:
            contributing = json.loads(str(row.get("contributing_agents", "[]")))
        except Exception:
            pass

        dx   = int(row.get("dx_found", 0))
        meds = int(row.get("meds_found", 0))
        clin = int(row.get("symptoms_found", 0))
        gt   = int(row["ground_truth"])
        pred = int(row["final_prediction"])

        rows.append({
            "note_id":          str(row.get("note_id", "")),
            "subject_id":       str(row.get("subject_id", "")),
            "ground_truth":     gt,
            "final_prediction": pred,
            "dx_found":         dx,
            "meds_found":       meds,
            "symptoms_found":   clin,
            "total_score":      int(row.get("total_score", 0)),
            "confidence":       str(row.get("confidence", "low")),
            "contributing_agents": contributing,
            "reason":           str(row.get("causal_chain") or row.get("reason", "")),
            "is_correct":       int(gt == pred),
            "found_count":      dx + meds + clin,
            "rule_pos":         int(dx == 1 or meds == 1),
            "llm_pos":          int(clin == 1),
            "is_disagreement":  int((dx == 1 or meds == 1) != (clin == 1)),
        })
    return rows


def select_featured_ids(df: pd.DataFrame) -> list:
    """Select up to 6 representative note_ids for the default card view."""
    featured = []
    used = set()

    def pick(mask, n):
        candidates = df[mask & ~df["note_id"].astype(str).isin(used)]
        ids = list(candidates.head(n)["note_id"].astype(str))
        used.update(ids)
        featured.extend(ids)

    fc = df["dx_found"] + df["meds_found"] + df["symptoms_found"]
    correct = df["ground_truth"] == df["final_prediction"]

    # Cat A: all 3 agents found, correct, high confidence
    pick((fc == 3) & correct & (df["confidence"] == "high"), 2)
    # Cat B: 1-2 agents, correct
    pick((fc <= 2) & correct, 2)
    # Cat C: wrong predictions
    pick(~correct, 2)
    # Fill remaining if < 6
    if len(featured) < min(6, len(df)):
        pick(pd.Series([True] * len(df), index=df.index), min(6, len(df)) - len(featured))

    return featured[:6]


def compute_comparison(df_c, df_a, df_b) -> list:
    entries = []
    for label, df in [("Horizontal (C)", df_c), ("Vertical (A)", df_a), ("Variant (B)", df_b)]:
        if df is not None and len(df) > 0:
            m = compute_metrics(df)
            entries.append({
                "label": label,
                "accuracy": m["accuracy"],
                "auc": m["auc"],
                "sensitivity": m["sensitivity"],
                "ppv": m["ppv"],
                "total": m["total"],
            })
    return entries


# ── HTML Template ──────────────────────────────────────────────────────────────
# Placeholders: INJECT_DATA, INJECT_METRICS, INJECT_AGENT_DEFS,
#               INJECT_AGENT_CONTRIB, INJECT_DISAGREEMENT, INJECT_COMPARISON,
#               INJECT_FEATURED_IDS, INJECT_TOTAL_RECORDS

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AD/ADRD Multi-Agent Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* ── Reset & Base ─────────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#2c3e50}
a{color:inherit}
/* ── Header ──────────────────────────────────────────────── */
header{background:linear-gradient(135deg,#1a237e,#283593);color:#fff;padding:24px 40px}
header h1{font-size:1.8rem}
header p{font-size:.95rem;opacity:.85;margin-top:4px}
/* ── Layout ──────────────────────────────────────────────── */
main{max-width:1300px;margin:0 auto;padding:32px 24px}
section{margin-bottom:56px}
h2{font-size:1.4rem;color:#1a237e;border-left:4px solid #3f51b5;
   padding-left:12px;margin-bottom:20px}
h3{font-size:1.05rem;color:#37474f;margin-bottom:14px}
.subtitle{font-size:.85rem;color:#78909c;font-weight:normal}
/* ── Filter / Search Bar ─────────────────────────────────── */
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:20px}
.filter-btn{padding:6px 16px;border-radius:20px;border:1.5px solid #c5cae9;
            background:#fff;color:#3f51b5;font-size:.82rem;font-weight:600;
            cursor:pointer;transition:all .15s}
.filter-btn:hover{background:#e8eaf6}
.filter-btn.active{background:#3f51b5;color:#fff;border-color:#3f51b5}
.search-box{padding:6px 14px;border-radius:20px;border:1.5px solid #c5cae9;
            font-size:.85rem;width:200px;outline:none;transition:border .15s}
.search-box:focus{border-color:#3f51b5}
.count-label{font-size:.82rem;color:#78909c;margin-left:auto}
/* ── Cards Grid ──────────────────────────────────────────── */
.cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:20px}
.card{background:#fff;border-radius:12px;border:1px solid #e0e0e0;
      box-shadow:0 2px 8px rgba(0,0,0,.07);overflow:hidden;transition:box-shadow .2s}
.card:hover{box-shadow:0 4px 16px rgba(0,0,0,.12)}
.card-top{padding:18px 18px 0}
/* ── Card Header ─────────────────────────────────────────── */
.card-header{display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.card-title{font-weight:700;font-size:.95rem;color:#1a237e}
.card-sub{font-size:.8rem;color:#90a4ae;flex:1}
/* ── Badges ──────────────────────────────────────────────── */
.agent-badges{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.badge{padding:4px 11px;border-radius:20px;font-size:.76rem;font-weight:600;
       display:inline-flex;align-items:center;gap:4px;white-space:nowrap}
.badge-yes{background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7}
.badge-no {background:#fce4ec;color:#c62828;border:1px solid #ef9a9a}
.badge-agent{cursor:pointer;transition:filter .15s;user-select:none}
.badge-agent:hover{filter:brightness(.93)}
.badge-info{font-size:.7rem;opacity:.7;margin-left:2px}
.result-correct{background:#e8f5e9;color:#1b5e20}
.result-wrong  {background:#fce4ec;color:#b71c1c}
/* ── Evidence Bar ────────────────────────────────────────── */
.bar-wrap{margin-bottom:12px}
.bar-label{font-size:.76rem;color:#78909c}
.bar-bg{background:#eceff1;border-radius:6px;height:7px;margin-top:4px}
.bar-fill{background:linear-gradient(90deg,#3f51b5,#7c4dff);border-radius:6px;
          height:7px;transition:width .3s}
/* ── Score / Prediction ──────────────────────────────────── */
.score-row,.pred-row{display:flex;justify-content:space-between;
                     font-size:.85rem;margin-bottom:7px}
.pred-pos{color:#c62828}
.pred-neg{color:#1565c0}
/* ── Expand Toggle ───────────────────────────────────────── */
.toggle-btn{width:100%;text-align:center;padding:8px;font-size:.78rem;
            color:#3f51b5;background:none;border:none;border-top:1px solid #f0f0f0;
            cursor:pointer;display:flex;align-items:center;justify-content:center;gap:4px}
.toggle-btn:hover{background:#f5f5f5}
.toggle-arrow{display:inline-block;transition:transform .2s;font-size:.7rem}
.card-detail{padding:14px 18px 16px;border-top:1px solid #f0f0f0;display:none}
.card-detail.open{display:block}
.reason-box{background:#f7f8fc;border-radius:6px;padding:10px 12px;
            font-size:.81rem;color:#455a64;line-height:1.55;
            border-left:3px solid #90a4ae;margin-bottom:10px}
.contrib-list{display:flex;flex-wrap:wrap;gap:6px;font-size:.78rem}
.contrib-chip{background:#e8eaf6;color:#3f51b5;border-radius:12px;
              padding:2px 10px;border:1px solid #c5cae9}
/* ── Load More ───────────────────────────────────────────── */
.load-more-wrap{text-align:center;margin-top:24px}
.load-more-btn{padding:10px 32px;border-radius:24px;border:2px solid #3f51b5;
               background:#fff;color:#3f51b5;font-size:.9rem;font-weight:600;
               cursor:pointer;transition:all .15s}
.load-more-btn:hover{background:#3f51b5;color:#fff}
.load-more-btn:disabled{opacity:.4;cursor:default}
/* ── Dashboard Metrics ───────────────────────────────────── */
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px}
.metric-box{background:#fff;border-radius:12px;padding:24px 16px;text-align:center;
            box-shadow:0 2px 8px rgba(0,0,0,.07)}
.metric-val{font-size:2.1rem;font-weight:700;color:#1a237e}
.metric-lbl{font-size:.82rem;color:#78909c;margin-top:4px}
/* ── Charts Row ──────────────────────────────────────────── */
.charts-row{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}
.chart-box{background:#fff;border-radius:12px;padding:22px;
           box-shadow:0 2px 8px rgba(0,0,0,.07)}
/* ── Confusion Matrix ────────────────────────────────────── */
.cm-wrap{display:flex;flex-direction:column;align-items:center;gap:6px}
.cm-header{display:grid;grid-template-columns:90px 1fr 1fr;width:100%;gap:4px}
.cm-header-cell{text-align:center;font-size:.75rem;color:#78909c;font-weight:600;padding:4px}
.cm-row{display:grid;grid-template-columns:90px 1fr 1fr;width:100%;gap:4px}
.cm-row-label{display:flex;align-items:center;justify-content:flex-end;
              padding-right:10px;font-size:.75rem;color:#78909c;font-weight:600}
.cm-cell{border-radius:10px;padding:14px 8px;text-align:center;min-height:70px;
         display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px}
.cm-tp,.cm-tn{background:#e8f5e9;border:1.5px solid #a5d6a7}
.cm-fp,.cm-fn{background:#fce4ec;border:1.5px solid #ef9a9a}
.cm-count{font-size:1.7rem;font-weight:700}
.cm-tp .cm-count,.cm-tn .cm-count{color:#2e7d32}
.cm-fp .cm-count,.cm-fn .cm-count{color:#c62828}
.cm-label{font-size:.72rem;font-weight:600;letter-spacing:.5px;opacity:.8}
.cm-pct{font-size:.75rem;color:#78909c}
.cm-axis-label{font-size:.78rem;font-weight:600;color:#37474f;margin-bottom:4px}
/* ── Disagreement Section ────────────────────────────────── */
.disagree-section{background:#fff;border-radius:12px;padding:22px;
                  box-shadow:0 2px 8px rgba(0,0,0,.07);margin-bottom:28px}
.disagree-summary{display:flex;gap:24px;margin-bottom:16px;flex-wrap:wrap}
.disagree-stat{background:#fff3e0;border-radius:8px;padding:12px 18px;
               border:1px solid #ffe0b2;text-align:center}
.disagree-stat .ds-val{font-size:1.5rem;font-weight:700;color:#e65100}
.disagree-stat .ds-lbl{font-size:.78rem;color:#78909c;margin-top:2px}
.disagree-table{width:100%;border-collapse:collapse;font-size:.83rem}
.disagree-table th{background:#f5f5f5;padding:8px 12px;text-align:left;
                   font-size:.78rem;color:#78909c;border-bottom:2px solid #e0e0e0}
.disagree-table td{padding:8px 12px;border-bottom:1px solid #f0f0f0}
.disagree-table tr:last-child td{border-bottom:none}
.disagree-table tr:hover td{background:#fafafa}
.tag-rule{background:#e3f2fd;color:#1565c0;border-radius:10px;
          padding:2px 8px;font-size:.73rem;font-weight:600}
.tag-llm{background:#f3e5f5;color:#6a1b9a;border-radius:10px;
         padding:2px 8px;font-size:.73rem;font-weight:600}
/* ── Comparison Table ────────────────────────────────────── */
.comparison-section{margin-top:8px}
.comparison-table{width:100%;border-collapse:collapse;background:#fff;
                  border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.07)}
.comparison-table th{background:#1a237e;color:#fff;padding:12px 16px;
                     font-size:.85rem;text-align:left}
.comparison-table td{padding:11px 16px;font-size:.85rem;border-bottom:1px solid #eceff1}
.comparison-table tr:last-child td{border-bottom:none}
.comparison-table tr:hover td{background:#f5f5f5}
/* ── Popup ───────────────────────────────────────────────── */
.popup-overlay{position:fixed;inset:0;background:rgba(0,0,0,.45);
               display:flex;align-items:center;justify-content:center;
               z-index:1000;opacity:0;pointer-events:none;transition:opacity .2s}
.popup-overlay.visible{opacity:1;pointer-events:all}
.popup{background:#fff;border-radius:14px;padding:0;max-width:480px;width:90%;
       max-height:80vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,.2);
       transform:scale(.95);transition:transform .2s}
.popup-overlay.visible .popup{transform:scale(1)}
.popup-head{display:flex;align-items:center;gap:10px;padding:16px 20px;
            border-bottom:1px solid #f0f0f0;position:sticky;top:0;background:#fff;
            border-radius:14px 14px 0 0;z-index:1}
.popup-icon{font-size:1.4rem}
.popup-title{font-weight:700;font-size:1rem;color:#1a237e;flex:1}
.popup-type-tag{font-size:.72rem;padding:3px 9px;border-radius:10px;font-weight:600}
.type-rule{background:#e3f2fd;color:#1565c0}
.type-llm {background:#f3e5f5;color:#6a1b9a}
.popup-close{background:none;border:none;font-size:1.3rem;cursor:pointer;
             color:#90a4ae;padding:0 4px;line-height:1;transition:color .15s}
.popup-close:hover{color:#e53935}
.popup-body{padding:16px 20px 20px}
.popup-status{display:flex;align-items:center;gap:8px;margin-bottom:14px;
              padding:10px 14px;border-radius:8px;font-weight:600;font-size:.88rem}
.popup-status.found   {background:#e8f5e9;color:#2e7d32}
.popup-status.notfound{background:#fce4ec;color:#c62828}
.popup-section{margin-bottom:12px}
.popup-section-title{font-size:.75rem;font-weight:700;color:#78909c;
                     text-transform:uppercase;letter-spacing:.6px;margin-bottom:5px}
.popup-section-body{font-size:.84rem;color:#455a64;line-height:1.55;
                    background:#f7f8fc;border-radius:6px;padding:9px 12px}
.popup-score-chip{display:inline-block;background:#e8eaf6;color:#3f51b5;
                  border-radius:10px;padding:3px 12px;font-size:.82rem;font-weight:700;
                  margin-top:4px}
/* ── Empty State ─────────────────────────────────────────── */
.empty-state{text-align:center;padding:48px 20px;color:#90a4ae;font-size:.95rem}
/* ── Responsive ──────────────────────────────────────────── */
@media(max-width:900px){
  .metrics-grid{grid-template-columns:repeat(2,1fr)}
  .charts-row{grid-template-columns:1fr}
  .cards-grid{grid-template-columns:1fr}
  header{padding:20px 20px}
  main{padding:20px 16px}
}
@media(max-width:480px){
  .metrics-grid{grid-template-columns:repeat(2,1fr)}
  .search-box{width:150px}
  .toolbar{gap:6px}
}
</style>
</head>
<body>

<header>
  <h1>AD/ADRD Identification &mdash; Multi-Agent System Report</h1>
  <p>Horizontal Architecture (main_c.py) &nbsp;&bull;&nbsp;
     <span id="total-count">INJECT_TOTAL_RECORDS</span> records evaluated</p>
</header>

<main>

<!-- ══ PART 1 ══════════════════════════════════════════════════════════════ -->
<section id="part1">
  <h2>Part 1: Decision Cards</h2>

  <!-- Toolbar -->
  <div class="toolbar">
    <button class="filter-btn active" data-filter="featured">Featured (6)</button>
    <button class="filter-btn" data-filter="all">All</button>
    <button class="filter-btn" data-filter="correct">Correct Only</button>
    <button class="filter-btn" data-filter="wrong">Wrong Only</button>
    <button class="filter-btn" data-filter="medium">Medium Confidence</button>
    <button class="filter-btn" data-filter="disagree">Agent Disagreement</button>
    <input class="search-box" type="text" id="note-search" placeholder="&#128269; Search note ID…">
    <span class="count-label" id="count-label">Showing 0 of 0</span>
  </div>

  <div class="cards-grid" id="cards-grid"></div>

  <div class="load-more-wrap" id="load-more-wrap" style="display:none">
    <button class="load-more-btn" id="load-more-btn">Load More</button>
  </div>
</section>

<!-- ══ PART 2 ══════════════════════════════════════════════════════════════ -->
<section id="part2">
  <h2>Part 2: Overall Performance Dashboard</h2>

  <!-- Big 4 metrics -->
  <div class="metrics-grid" id="metrics-grid"></div>

  <!-- Confusion Matrix + Grouped Bar Chart -->
  <div class="charts-row">
    <div class="chart-box">
      <h3>Confusion Matrix</h3>
      <div id="confusion-matrix"></div>
    </div>
    <div class="chart-box">
      <h3>Agent Contribution Analysis</h3>
      <p style="font-size:.78rem;color:#90a4ae;margin-bottom:12px">
        Flag rate: % records where agent flagged positive &nbsp;&bull;&nbsp;
        Precision: % of flagged records that are truly AD/ADRD+
      </p>
      <canvas id="contrib-chart" height="200"></canvas>
    </div>
  </div>

  <!-- Disagreement Section -->
  <div class="disagree-section">
    <h3>Agent Disagreement Cases
      <span class="subtitle">— Rule agents (ICD+Meds) vs LLM agent (ClinText) give conflicting signals</span>
    </h3>
    <div id="disagree-content"></div>
  </div>

  <!-- System Comparison (shown only if predictions_a.csv or _b.csv exist) -->
  <div id="comparison-section"></div>

</section>

</main>

<!-- ══ POPUP OVERLAY ═══════════════════════════════════════════════════════ -->
<div class="popup-overlay" id="popup-overlay" role="dialog" aria-modal="true">
  <div class="popup" id="popup">
    <div class="popup-head">
      <span class="popup-icon" id="popup-icon"></span>
      <span class="popup-title" id="popup-title"></span>
      <span class="popup-type-tag" id="popup-type-tag"></span>
      <button class="popup-close" id="popup-close" aria-label="Close">&times;</button>
    </div>
    <div class="popup-body" id="popup-body"></div>
  </div>
</div>

<script>
// ── Injected data ──────────────────────────────────────────────────────────
const DATA          = INJECT_DATA;
const METRICS       = INJECT_METRICS;
const AGENT_DEFS    = INJECT_AGENT_DEFS;
const AGENT_CONTRIB = INJECT_AGENT_CONTRIB;
const DISAGREEMENT  = INJECT_DISAGREEMENT;
const COMPARISON    = INJECT_COMPARISON;
const FEATURED_IDS  = INJECT_FEATURED_IDS;

// ── State ──────────────────────────────────────────────────────────────────
const PAGE_SIZE = 20;
let currentFilter = 'featured';
let currentSearch = '';
let visibleCount  = 6;

// ── Utility ────────────────────────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function confColor(c) {
  return {high:'#2e7d32', medium:'#e65100', low:'#c62828'}[c] || '#78909c';
}

// ── Filter Logic ───────────────────────────────────────────────────────────
function getFiltered() {
  let rows = [...DATA];
  const q = currentSearch.trim().toLowerCase();
  if (q) rows = rows.filter(r => r.note_id.toLowerCase().includes(q));

  switch (currentFilter) {
    case 'featured': return rows.filter(r => FEATURED_IDS.includes(r.note_id));
    case 'correct':  return rows.filter(r => r.is_correct);
    case 'wrong':    return rows.filter(r => !r.is_correct);
    case 'medium':   return rows.filter(r => r.confidence === 'medium');
    case 'disagree': return rows.filter(r => r.is_disagreement);
    default:         return rows;
  }
}

// ── Card Rendering ─────────────────────────────────────────────────────────
function buildBadge(row, agDef) {
  const found = row[agDef.key];
  const cls   = found ? 'badge-yes' : 'badge-no';
  const icon  = found ? '&#10003;' : '&#10007;';
  return `<span class="badge ${cls} badge-agent"
    data-note-id="${esc(row.note_id)}" data-agent-key="${esc(agDef.key)}"
    title="Click for details">${icon} ${esc(agDef.label)} <span class="badge-info">&#9432;</span></span>`;
}

function buildCard(row) {
  const isCorrect = row.is_correct;
  const gt    = row.ground_truth === 1 ? 'AD/ADRD+' : 'No ADRD';
  const pred  = row.final_prediction === 1 ? 'AD/ADRD+' : 'No ADRD';
  const barPct = Math.round(row.found_count / 4 * 100);
  const contribs = (row.contributing_agents || [])
    .map(a => `<span class="contrib-chip">${esc(a)}</span>`).join('');

  const badges = AGENT_DEFS.map(a => buildBadge(row, a)).join('');

  return `
<div class="card" data-note-id="${esc(row.note_id)}">
  <div class="card-top">
    <div class="card-header">
      <span class="card-title">Note ID: ${esc(row.note_id)}</span>
      <span class="card-sub">Subject: ${esc(row.subject_id)}</span>
      <span class="badge ${isCorrect ? 'result-correct' : 'result-wrong'}">
        ${isCorrect ? '&#10003; CORRECT' : '&#10007; WRONG'}
      </span>
    </div>
    <div class="agent-badges">${badges}</div>
    <div class="bar-wrap">
      <span class="bar-label">Evidence Strength (${row.found_count}/4 agents positive)</span>
      <div class="bar-bg"><div class="bar-fill" style="width:${barPct}%"></div></div>
    </div>
    <div class="score-row">
      <span>Total Score: <strong>${row.total_score}/9</strong></span>
      <span style="color:${confColor(row.confidence)}">
        Confidence: <strong>${row.confidence.toUpperCase()}</strong>
      </span>
    </div>
    <div class="pred-row">
      <span>Prediction: <strong class="${row.final_prediction===1?'pred-pos':'pred-neg'}">${pred}</strong></span>
      <span>Ground Truth: <strong>${gt}</strong></span>
    </div>
  </div>
  <button class="toggle-btn" data-note-id="${esc(row.note_id)}">
    <span class="toggle-arrow">&#9660;</span>&nbsp;Details
  </button>
  <div class="card-detail" id="detail-${esc(row.note_id)}">
    <div class="reason-box">${esc(row.reason)}</div>
    ${contribs ? `<div style="margin-top:8px;font-size:.78rem;color:#78909c;margin-bottom:4px">Contributing agents:</div>
    <div class="contrib-list">${contribs}</div>` : ''}
    ${row.is_disagreement ? `<div style="margin-top:10px;font-size:.78rem;padding:6px 10px;background:#fff3e0;border-radius:6px;color:#e65100;border-left:3px solid #ff9800">
      &#9888; Agent Disagreement: rule-based and LLM agents gave conflicting signals
    </div>` : ''}
  </div>
</div>`;
}

function renderCards() {
  const filtered = getFiltered();
  const grid     = document.getElementById('cards-grid');
  const countLbl = document.getElementById('count-label');
  const moreWrap = document.getElementById('load-more-wrap');
  const moreBtn  = document.getElementById('load-more-btn');

  if (filtered.length === 0) {
    grid.innerHTML = '<div class="empty-state">No records match the current filter.</div>';
    countLbl.textContent = 'Showing 0 of 0';
    moreWrap.style.display = 'none';
    return;
  }

  const isFeatured = (currentFilter === 'featured' && !currentSearch.trim());
  const showCount  = isFeatured ? filtered.length : Math.min(visibleCount, filtered.length);
  const displayed  = filtered.slice(0, showCount);

  grid.innerHTML = displayed.map(buildCard).join('');
  countLbl.textContent = `Showing ${displayed.length} of ${filtered.length} records`;

  if (!isFeatured && filtered.length > showCount) {
    moreWrap.style.display = 'block';
    moreBtn.disabled = false;
    moreBtn.textContent = `Load More (${Math.min(PAGE_SIZE, filtered.length - showCount)} more)`;
  } else {
    moreWrap.style.display = 'none';
  }
}

// ── Popup Logic ────────────────────────────────────────────────────────────
function showPopup(noteId, agentKey) {
  const row     = DATA.find(r => r.note_id === noteId);
  const agDef   = AGENT_DEFS.find(a => a.key === agentKey);
  if (!row || !agDef) return;

  const found      = row[agentKey];
  const scoreAdd   = found ? agDef.score : 0;
  const isRule     = agDef.agent_type.toLowerCase().includes('rule');
  const typeClass  = isRule ? 'type-rule' : 'type-llm';
  const icon       = found ? '&#9989;' : '&#10060;';
  const statusCls  = found ? 'found' : 'notfound';
  const statusTxt  = found ? 'Evidence Found' : 'No Evidence Found';
  const evidenceTxt = esc(found ? agDef.found_msg : agDef.not_found_msg);

  document.getElementById('popup-icon').innerHTML    = icon;
  document.getElementById('popup-title').textContent = agDef.name;
  const typeTag = document.getElementById('popup-type-tag');
  typeTag.textContent  = agDef.agent_type;
  typeTag.className    = `popup-type-tag ${typeClass}`;

  document.getElementById('popup-body').innerHTML = `
    <div class="popup-status ${statusCls}">${icon}&nbsp;${statusTxt}
      &nbsp;<span class="popup-score-chip">+${scoreAdd} pts</span>
    </div>
    <div class="popup-section">
      <div class="popup-section-title">Methodology</div>
      <div class="popup-section-body">${esc(agDef.method)}</div>
    </div>
    <div class="popup-section">
      <div class="popup-section-title">Finding</div>
      <div class="popup-section-body">${evidenceTxt}</div>
    </div>
    <div class="popup-section">
      <div class="popup-section-title">Context &mdash; Note ${esc(noteId)}</div>
      <div class="popup-section-body">
        Overall score: <strong>${row.total_score}/9</strong> &nbsp;&bull;&nbsp;
        Confidence: <strong style="color:${confColor(row.confidence)}">${row.confidence.toUpperCase()}</strong><br>
        Final prediction: <strong>${row.final_prediction===1?'AD/ADRD+':'No ADRD'}</strong> &nbsp;&bull;&nbsp;
        Ground truth: <strong>${row.ground_truth===1?'AD/ADRD+':'No ADRD'}</strong>
        ${row.is_disagreement ? '<br><span style="color:#e65100;font-size:.8rem">&#9888; Rule vs LLM agents disagreed on this record</span>' : ''}
      </div>
    </div>
    <div class="popup-section">
      <div class="popup-section-title">Synthesizer Reasoning</div>
      <div class="popup-section-body">${esc(row.reason)}</div>
    </div>`;

  document.getElementById('popup-overlay').classList.add('visible');
  document.getElementById('popup-close').focus();
}

function closePopup() {
  document.getElementById('popup-overlay').classList.remove('visible');
}

// ── Confusion Matrix ───────────────────────────────────────────────────────
function renderConfusionMatrix() {
  const m = METRICS;
  const pct = n => m.total > 0 ? (n/m.total*100).toFixed(1)+'%' : '0%';
  document.getElementById('confusion-matrix').innerHTML = `
    <div class="cm-wrap">
      <div style="font-size:.78rem;color:#78909c;font-weight:600;margin-bottom:2px;text-align:center">
        Predicted Label
      </div>
      <div class="cm-header">
        <div></div>
        <div class="cm-header-cell">Predicted +</div>
        <div class="cm-header-cell">Predicted &minus;</div>
      </div>
      <div class="cm-row">
        <div class="cm-row-label">Actual +</div>
        <div class="cm-cell cm-tp">
          <div class="cm-count">${m.tp}</div>
          <div class="cm-label">TP</div>
          <div class="cm-pct">${pct(m.tp)}</div>
        </div>
        <div class="cm-cell cm-fn">
          <div class="cm-count">${m.fn}</div>
          <div class="cm-label">FN</div>
          <div class="cm-pct">${pct(m.fn)}</div>
        </div>
      </div>
      <div class="cm-row">
        <div class="cm-row-label">Actual &minus;</div>
        <div class="cm-cell cm-fp">
          <div class="cm-count">${m.fp}</div>
          <div class="cm-label">FP</div>
          <div class="cm-pct">${pct(m.fp)}</div>
        </div>
        <div class="cm-cell cm-tn">
          <div class="cm-count">${m.tn}</div>
          <div class="cm-label">TN</div>
          <div class="cm-pct">${pct(m.tn)}</div>
        </div>
      </div>
      <div style="font-size:.73rem;color:#90a4ae;margin-top:8px;text-align:center">
        N = ${m.total} &nbsp;&bull;&nbsp; Accuracy = ${m.accuracy}
      </div>
    </div>`;
}

// ── Grouped Bar Chart ──────────────────────────────────────────────────────
function renderGroupedBar() {
  const ctx    = document.getElementById('contrib-chart').getContext('2d');
  const labels = AGENT_CONTRIB.map(a => a.label);
  const rates  = AGENT_CONTRIB.map(a => a.flag_rate);
  const precs  = AGENT_CONTRIB.map(a => a.precision);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Flag Rate (%)',       data: rates, backgroundColor: 'rgba(63,81,181,.75)',
          borderColor: '#3f51b5', borderWidth: 1.5, borderRadius: 4 },
        { label: 'Precision When Flagged (%)', data: precs, backgroundColor: 'rgba(46,125,50,.7)',
          borderColor: '#2e7d32', borderWidth: 1.5, borderRadius: 4 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { position: 'bottom', labels: { font: { size: 11 } } },
                 tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%` } } },
      scales: {
        y: { min:0, max:100, ticks: { callback: v => v+'%', font:{size:11} },
             grid: { color:'rgba(0,0,0,.06)' } },
        x: { ticks: { font:{size:11} }, grid: { display:false } },
      },
    },
  });
}

// ── Big Metrics ────────────────────────────────────────────────────────────
function renderMetrics() {
  const m = METRICS;
  document.getElementById('metrics-grid').innerHTML = [
    ['Accuracy',         m.accuracy],
    ['AUC',              m.auc],
    ['Sensitivity',      m.sensitivity],
    ['PPV (Precision)',  m.ppv],
  ].map(([lbl, val]) => `
    <div class="metric-box">
      <div class="metric-val">${val}</div>
      <div class="metric-lbl">${lbl}</div>
    </div>`).join('');
}

// ── Disagreement Section ───────────────────────────────────────────────────
function renderDisagreement() {
  const d = DISAGREEMENT;
  const el = document.getElementById('disagree-content');
  if (d.total === 0) {
    el.innerHTML = '<p style="font-size:.87rem;color:#78909c;">No disagreement cases found — all rule and LLM agents agreed on every record.</p>';
    return;
  }
  const rows = d.cases.map(c => `
    <tr>
      <td>${esc(c.note_id)}</td>
      <td><span class="tag-rule">${c.dx_found?'ICD&#10003;':'ICD&#10007;'}</span>
          <span class="tag-rule">${c.meds_found?'Meds&#10003;':'Meds&#10007;'}</span></td>
      <td><span class="tag-llm">${c.symptoms_found?'Clin&#10003;':'Clin&#10007;'}</span></td>
      <td>${c.final_prediction===1?'<span class="pred-pos">AD/ADRD+</span>':'<span class="pred-neg">No ADRD</span>'}</td>
      <td>${c.ground_truth===1?'AD/ADRD+':'No ADRD'}</td>
      <td><span class="badge ${c.is_correct?'badge-yes':'badge-no'}">${c.is_correct?'&#10003; Correct':'&#10007; Wrong'}</span></td>
    </tr>`).join('');

  el.innerHTML = `
    <div class="disagree-summary">
      <div class="disagree-stat">
        <div class="ds-val">${d.total}</div>
        <div class="ds-lbl">Disagreement Cases</div>
      </div>
      <div class="disagree-stat" style="background:#f3e5f5;border-color:#ce93d8">
        <div class="ds-val" style="color:#6a1b9a">${d.accuracy}%</div>
        <div class="ds-lbl">Accuracy on Disagreements</div>
      </div>
      <div class="disagree-stat" style="background:#e3f2fd;border-color:#90caf9">
        <div class="ds-val" style="color:#1565c0">${Math.round(d.total/METRICS.total*100)}%</div>
        <div class="ds-lbl">of All Records</div>
      </div>
    </div>
    <table class="disagree-table">
      <thead>
        <tr>
          <th>Note ID</th><th>Rule Agents</th><th>LLM Agents</th>
          <th>Prediction</th><th>Ground Truth</th><th>Result</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ── System Comparison ──────────────────────────────────────────────────────
function renderComparison() {
  if (!COMPARISON || COMPARISON.length < 2) return;
  const rows = COMPARISON.map(c => `
    <tr>
      <td><strong>${esc(c.label)}</strong></td>
      <td>${c.accuracy}</td><td>${c.auc}</td>
      <td>${c.sensitivity}</td><td>${c.ppv}</td><td>${c.total}</td>
    </tr>`).join('');

  document.getElementById('comparison-section').innerHTML = `
    <div class="comparison-section">
      <h3>System Comparison: Horizontal vs Vertical</h3>
      <table class="comparison-table">
        <thead>
          <tr><th>System</th><th>Accuracy</th><th>AUC</th>
              <th>Sensitivity</th><th>PPV</th><th>N</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

// ── Filter button counts ───────────────────────────────────────────────────
function updateFilterCounts() {
  const filters = {
    featured: FEATURED_IDS.length,
    all:      DATA.length,
    correct:  DATA.filter(r => r.is_correct).length,
    wrong:    DATA.filter(r => !r.is_correct).length,
    medium:   DATA.filter(r => r.confidence==='medium').length,
    disagree: DATA.filter(r => r.is_disagreement).length,
  };
  document.querySelectorAll('.filter-btn').forEach(btn => {
    const f = btn.dataset.filter;
    if (f in filters) btn.textContent = btn.textContent.replace(/\s*\(.*\)/, '') + ` (${filters[f]})`;
  });
}

// ── Event Listeners ────────────────────────────────────────────────────────
function setupEvents() {
  // Filter buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.filter;
      visibleCount  = PAGE_SIZE;
      renderCards();
    });
  });

  // Search
  document.getElementById('note-search').addEventListener('input', e => {
    currentSearch = e.target.value;
    visibleCount  = PAGE_SIZE;
    if (currentSearch.trim()) {
      // Switch to "all" filter when searching
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      document.querySelector('[data-filter="all"]').classList.add('active');
      currentFilter = 'all';
    }
    renderCards();
  });

  // Load More
  document.getElementById('load-more-btn').addEventListener('click', () => {
    visibleCount += PAGE_SIZE;
    renderCards();
  });

  // Badge click → popup (event delegation)
  document.getElementById('cards-grid').addEventListener('click', e => {
    const badge = e.target.closest('.badge-agent');
    if (badge) {
      e.stopPropagation();
      showPopup(badge.dataset.noteId, badge.dataset.agentKey);
    }
    const toggle = e.target.closest('.toggle-btn');
    if (toggle) {
      const nid    = toggle.dataset.noteId;
      const detail = document.getElementById('detail-' + nid);
      const arrow  = toggle.querySelector('.toggle-arrow');
      if (detail) {
        detail.classList.toggle('open');
        arrow.style.transform = detail.classList.contains('open') ? 'rotate(180deg)' : '';
      }
    }
  });

  // Close popup
  document.getElementById('popup-close').addEventListener('click', closePopup);
  document.getElementById('popup-overlay').addEventListener('click', e => {
    if (e.target === document.getElementById('popup-overlay')) closePopup();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closePopup(); });
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  updateFilterCounts();
  renderCards();
  renderMetrics();
  renderConfusionMatrix();
  renderGroupedBar();
  renderDisagreement();
  renderComparison();
  setupEvents();
});
</script>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_c = load_predictions(OUTPUT_DIR / "predictions_c.csv")
    df_a = load_predictions(OUTPUT_DIR / "predictions_a.csv")
    df_b = load_predictions(OUTPUT_DIR / "predictions_b.csv")

    if df_c is None or len(df_c) == 0:
        print("[visualize] No valid results in predictions_c.csv. Run main_c.py first.")
        placeholder = HTML_TEMPLATE
        placeholder = placeholder.replace("INJECT_TOTAL_RECORDS", "0")
        placeholder = placeholder.replace("INJECT_DATA", "[]")
        placeholder = placeholder.replace("INJECT_METRICS", json.dumps({
            "accuracy": "N/A", "sensitivity": "N/A", "ppv": "N/A", "auc": "N/A",
            "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "tp_pct": 0, "fp_pct": 0, "fn_pct": 0, "tn_pct": 0, "correct": 0,
        }))
        placeholder = placeholder.replace("INJECT_AGENT_DEFS", json.dumps(AGENT_DEFS))
        placeholder = placeholder.replace("INJECT_AGENT_CONTRIB", "[]")
        placeholder = placeholder.replace("INJECT_DISAGREEMENT", json.dumps({"total": 0, "accuracy": 0, "cases": []}))
        placeholder = placeholder.replace("INJECT_COMPARISON", "[]")
        placeholder = placeholder.replace("INJECT_FEATURED_IDS", "[]")
        REPORT_PATH.write_text(placeholder, encoding="utf-8")
        print(f"[visualize] Placeholder saved to: {REPORT_PATH}")
        return

    # Compute all data
    all_rows    = serialize_rows(df_c)
    metrics     = compute_metrics(df_c)
    agent_contrib = compute_agent_contrib(df_c)
    disagreement = compute_disagreement_data(df_c)
    comparison   = compute_comparison(df_c, df_a, df_b)
    featured_ids = select_featured_ids(df_c)

    print(f"[visualize] Total records: {len(df_c)}")
    print(f"[visualize] Featured IDs: {featured_ids}")
    print(f"[visualize] Metrics: accuracy={metrics['accuracy']} auc={metrics['auc']} "
          f"sensitivity={metrics['sensitivity']} ppv={metrics['ppv']}")
    print(f"[visualize] Confusion matrix: TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TN={metrics['tn']}")
    print(f"[visualize] Disagreements: {disagreement['total']} cases")

    html = HTML_TEMPLATE
    html = html.replace("INJECT_TOTAL_RECORDS", str(len(df_c)))
    html = html.replace("INJECT_DATA",          json.dumps(all_rows,      ensure_ascii=False))
    html = html.replace("INJECT_METRICS",       json.dumps(metrics,       ensure_ascii=False))
    html = html.replace("INJECT_AGENT_DEFS",    json.dumps(AGENT_DEFS,    ensure_ascii=False))
    html = html.replace("INJECT_AGENT_CONTRIB", json.dumps(agent_contrib, ensure_ascii=False))
    html = html.replace("INJECT_DISAGREEMENT",  json.dumps(disagreement,  ensure_ascii=False))
    html = html.replace("INJECT_COMPARISON",    json.dumps(comparison,    ensure_ascii=False))
    html = html.replace("INJECT_FEATURED_IDS",  json.dumps(featured_ids,  ensure_ascii=False))

    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"[visualize] Report saved to: {REPORT_PATH}")
    print(f"            Open: file:///{REPORT_PATH.as_posix()}")


if __name__ == "__main__":
    main()
