"""
eval/report_generator.py — Self-contained HTML report for the AD/ADRD eval harness.

No external CSS/JS dependencies. Readable on any browser without internet access.
"""

import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _e(text) -> str:
    """HTML-escape a value."""
    return html.escape(str(text) if text is not None else "")


def _trunc(text, n: int = 400) -> str:
    s = str(text) if text else ""
    return s[:n] + "…" if len(s) > n else s


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{100 * n / total:.1f}%"


def _score_badge(score: float) -> str:
    """Colour-coded score badge."""
    if score >= 0.8:
        colour = "#27ae60"
    elif score >= 0.5:
        colour = "#f39c12"
    else:
        colour = "#e74c3c"
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 7px;'
        f'border-radius:3px;font-size:0.85em;">{score:.2f}</span>'
    )


# ─── Section builders ─────────────────────────────────────────────────────────

def _metrics_section(m: dict) -> str:
    rows = [
        ("Total labeled",   m["total"],                   ""),
        ("Correct",         m["correct"],                 _pct(m["correct"], m["total"])),
        ("Accuracy",        f"{m['accuracy']:.4f}",       ""),
        ("Sensitivity",     f"{m['sensitivity']:.4f}",    "(recall for AD/ADRD cases)"),
        ("PPV (Precision)", f"{m['ppv']:.4f}",            ""),
        ("Specificity",     f"{m['specificity']:.4f}",    ""),
        ("F1",              f"{m['f1']:.4f}",             ""),
    ]
    trs = "\n".join(
        f"<tr><td>{_e(k)}</td><td><strong>{_e(v)}</strong></td><td style='color:#666;font-size:0.9em'>{_e(note)}</td></tr>"
        for k, v, note in rows
    )
    return f"""
<section>
<h2>Overall Metrics</h2>
<table class="metrics-table">
<thead><tr><th>Metric</th><th>Value</th><th>Note</th></tr></thead>
<tbody>{trs}</tbody>
</table>
</section>"""


def _confusion_section(m: dict) -> str:
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    total = tp + fp + fn + tn
    return f"""
<section>
<h2>Confusion Matrix</h2>
<table class="confusion">
<thead>
  <tr><th></th><th colspan="2">Predicted</th></tr>
  <tr><th>Actual</th><th>Positive (1)</th><th>Negative (0)</th></tr>
</thead>
<tbody>
  <tr>
    <th>Positive (1)</th>
    <td class="tp">TP = {tp}<br><span class="sub">{_pct(tp, total)} of all</span></td>
    <td class="fn">FN = {fn}<br><span class="sub">{_pct(fn, total)} of all</span></td>
  </tr>
  <tr>
    <th>Negative (0)</th>
    <td class="fp">FP = {fp}<br><span class="sub">{_pct(fp, total)} of all</span></td>
    <td class="tn">TN = {tn}<br><span class="sub">{_pct(tn, total)} of all</span></td>
  </tr>
</tbody>
</table>
</section>"""


def _bucket_summary_section(buckets: dict, total: int) -> str:
    total_errors = len(buckets["false_positives"]) + len(buckets["false_negatives"])
    rows = [
        ("False Positives (FP)",      buckets["false_positives"],      "Predicted AD, actually negative"),
        ("False Negatives (FN)",      buckets["false_negatives"],      "Predicted No-AD, actually positive"),
        ("Contradictory errors",      buckets["contradictory_errors"], "Wrong prediction + agent discrepancy"),
        ("Low-confidence errors",     buckets["low_confidence_errors"],"Wrong prediction + confidence=low"),
        ("Consensus errors",          buckets["consensus_errors"],     "Wrong prediction on consensus path"),
    ]
    trs = "\n".join(
        f"<tr><td>{_e(label)}</td><td>{len(cases)}</td>"
        f"<td>{_pct(len(cases), total_errors) if total_errors else '-'}</td>"
        f"<td style='color:#666;font-size:0.9em'>{_e(note)}</td></tr>"
        for label, cases, note in rows
    )
    return f"""
<section>
<h2>Error Bucket Summary</h2>
<p>Total errors: <strong>{total_errors}</strong> / {total} labeled cases
   ({_pct(total_errors, total)})</p>
<table class="metrics-table">
<thead><tr><th>Bucket</th><th>Count</th><th>% of errors</th><th>Definition</th></tr></thead>
<tbody>{trs}</tbody>
</table>
</section>"""


def _case_row(row: dict, show_discrepancy: bool = False) -> str:
    note_id  = _e(row.get("note_id", ""))
    gt       = _e(row.get("ground_truth", ""))
    pred     = _e(row.get("final_prediction", ""))
    conf     = _e(row.get("confidence", ""))
    subtype  = _e(row.get("subtype", ""))
    mode     = _e(row.get("synthesis_mode", "—"))
    reason   = _e(_trunc(row.get("reason", ""), 200))
    chain    = _e(_trunc(row.get("causal_chain", ""), 500))
    disc     = _e(_trunc(row.get("discrepancy", ""), 400))

    detail = f"<div class='chain'><strong>Reasoning:</strong> {chain}</div>"
    if show_discrepancy and row.get("discrepancy", "").strip():
        detail += f"<div class='disc'><strong>Discrepancy:</strong> {disc}</div>"

    return f"""
<tr>
  <td class="mono">{note_id}</td>
  <td class="center">{gt}</td>
  <td class="center">{pred}</td>
  <td class="center">{conf}</td>
  <td class="center">{subtype}</td>
  <td class="center">{mode}</td>
  <td>{reason}</td>
</tr>
<tr class="detail-row"><td colspan="7">{detail}</td></tr>"""


def _case_table(cases: list[dict], show_discrepancy: bool = False) -> str:
    if not cases:
        return "<p class='empty'>No cases in this bucket.</p>"
    rows = "\n".join(_case_row(r, show_discrepancy) for r in cases)
    return f"""
<table class="case-table">
<thead>
  <tr>
    <th>note_id</th><th>GT</th><th>Pred</th>
    <th>Conf</th><th>Subtype</th><th>Mode</th><th>Reason</th>
  </tr>
</thead>
<tbody>{rows}</tbody>
</table>"""


def _fp_fn_section(buckets: dict) -> str:
    fp_table = _case_table(buckets["false_positives"])
    fn_table = _case_table(buckets["false_negatives"])
    return f"""
<section>
<h2>False Positives ({len(buckets["false_positives"])})</h2>
<details>
  <summary>Show {len(buckets["false_positives"])} FP cases</summary>
  {fp_table}
</details>
</section>

<section>
<h2>False Negatives ({len(buckets["false_negatives"])})</h2>
<details>
  <summary>Show {len(buckets["false_negatives"])} FN cases</summary>
  {fn_table}
</details>
</section>"""


def _contradictory_section(buckets: dict) -> str:
    all_cases = buckets["all_contradictory"]
    if not all_cases:
        return "<section><h2>Contradictory Cases</h2><p class='empty'>None found.</p></section>"

    error_ids = {r.get("note_id") for r in buckets["contradictory_errors"]}
    table = _case_table(all_cases, show_discrepancy=True)

    return f"""
<section>
<h2>Contradictory Cases — All {len(all_cases)} (errors highlighted)</h2>
<p>Cases where agents disagreed (discrepancy field non-empty).
   <strong>{len(error_ids)}</strong> of these were also prediction errors.</p>
<details open>
  <summary>Show all {len(all_cases)} contradictory cases</summary>
  {table}
</details>
</section>"""


def _judge_section(judge_results: list[dict]) -> str:
    if not judge_results:
        return ""

    avg_overall = sum(r["overall_score"] for r in judge_results) / len(judge_results)
    avg_clarity = sum(r["reasoning_clarity"] for r in judge_results) / len(judge_results)
    avg_contr   = sum(r["contradiction_handling"] for r in judge_results) / len(judge_results)
    avg_evid    = sum(r["evidence_consistency"] for r in judge_results) / len(judge_results)

    rows = []
    for r in judge_results:
        correct_tag = (
            '<span style="color:#27ae60">✓</span>' if r.get("is_correct")
            else '<span style="color:#e74c3c">✗</span>'
        )
        rows.append(f"""
<tr>
  <td class="mono">{_e(r["note_id"])}</td>
  <td class="center">{_e(r["ground_truth"])}</td>
  <td class="center">{_e(r["final_prediction"])}</td>
  <td class="center">{correct_tag}</td>
  <td class="center">{_score_badge(r["reasoning_clarity"])}</td>
  <td class="center">{_score_badge(r["contradiction_handling"])}</td>
  <td class="center">{_score_badge(r["evidence_consistency"])}</td>
  <td class="center">{_score_badge(r["overall_score"])}</td>
  <td>{_e(_trunc(r["judge_comment"], 300))}</td>
</tr>""")

    summary = f"""
<table class="metrics-table" style="margin-bottom:1em">
<tr>
  <td>Avg Overall</td><td>{_score_badge(avg_overall)}</td>
  <td>Avg Clarity</td><td>{_score_badge(avg_clarity)}</td>
  <td>Avg Contradiction</td><td>{_score_badge(avg_contr)}</td>
  <td>Avg Evidence</td><td>{_score_badge(avg_evid)}</td>
</tr>
</table>"""

    table = f"""
<table class="case-table">
<thead>
  <tr>
    <th>note_id</th><th>GT</th><th>Pred</th><th>Correct</th>
    <th>Clarity</th><th>Contradiction</th><th>Evidence</th><th>Overall</th><th>Comment</th>
  </tr>
</thead>
<tbody>{"".join(rows)}</tbody>
</table>"""

    return f"""
<section>
<h2>LLM Judge Results ({len(judge_results)} cases)</h2>
{summary}
<details open>
  <summary>Show detailed judge scores</summary>
  {table}
</details>
</section>"""


def _confidence_dist_section(df: pd.DataFrame) -> str:
    """Simple bar chart using inline CSS — no JS needed."""
    needed = ["confidence_score_clin", "confidence_score_med", "confidence_score_dx"]
    if not all(c in df.columns for c in needed):
        return ""
    if df[needed].sum().sum() == 0:
        return ""

    def _bar_row(label: str, series: "pd.Series") -> str:
        buckets_def = [
            ("0.0–0.2", 0.0, 0.2, "#e74c3c"),
            ("0.2–0.5", 0.2, 0.5, "#f39c12"),
            ("0.5–0.8", 0.5, 0.8, "#3498db"),
            ("0.8–1.0", 0.8, 1.0, "#27ae60"),
        ]
        cells = ""
        for name, lo, hi, colour in buckets_def:
            count = ((series >= lo) & (series < hi)).sum()
            pct = 100 * count / len(series) if len(series) else 0
            cells += (
                f"<td style='padding:4px 8px;'>"
                f"<div style='background:{colour};width:{max(pct,2):.0f}px;height:16px;"
                f"display:inline-block;vertical-align:middle'></div> "
                f"<span style='font-size:0.85em'>{name}: {count} ({pct:.0f}%)</span></td>"
            )
        return f"<tr><td><strong>{_e(label)}</strong></td>{cells}</tr>"

    rows = (
        _bar_row("ClinTextAgent",   df["confidence_score_clin"]) +
        _bar_row("MedicationAgent", df["confidence_score_med"]) +
        _bar_row("DiagnosisAgent",  df["confidence_score_dx"])
    )

    return f"""
<section>
<h2>Agent Confidence Score Distribution</h2>
<p style="color:#666;font-size:0.9em">Rule-based scores (0–1) from each specialist agent.
   Requires Stage 8 outputs; shown only when data is available.</p>
<table><tbody>{rows}</tbody></table>
</section>"""


# ─── CSS ─────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       font-size: 14px; color: #333; background: #f8f9fa; padding: 20px; }
h1   { font-size: 1.6em; margin-bottom: 4px; }
h2   { font-size: 1.15em; color: #2c3e50; margin: 24px 0 10px; border-bottom: 2px solid #3498db; padding-bottom: 4px; }
p    { margin: 6px 0; }
section { background: #fff; border-radius: 6px; padding: 16px 20px; margin-bottom: 16px;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.header-meta { color: #666; font-size: 0.9em; margin-top: 2px; }

/* Tables */
table { border-collapse: collapse; width: 100%; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }
th { background: #f0f4f8; font-weight: 600; }
.metrics-table td:nth-child(2) { font-weight: bold; }
.center { text-align: center; }
.mono   { font-family: monospace; font-size: 0.88em; }
.sub    { font-size: 0.82em; color: #666; }

/* Confusion matrix */
.confusion th { background: #ecf0f1; }
.confusion td { text-align: center; padding: 12px 20px; font-size: 1em; }
.tp { background: #d5f5e3; }
.tn { background: #d5f5e3; }
.fp { background: #fde8e8; }
.fn { background: #fde8e8; }

/* Case table */
.case-table { font-size: 0.85em; }
.case-table tr:hover td { background: #f0f7ff; }
.detail-row td { background: #fafbfc; padding: 6px 12px; }
.chain { color: #444; margin-bottom: 4px; }
.disc  { color: #8e44ad; margin-top: 4px; }

/* Details/Summary */
details { margin-top: 8px; }
summary { cursor: pointer; color: #2980b9; font-weight: 500; padding: 4px 0; }
summary:hover { text-decoration: underline; }
.empty { color: #888; font-style: italic; padding: 8px 0; }
"""


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_report(
    metrics: dict,
    buckets: dict,
    judge_results: list[dict],
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Build and write the HTML report. Returns the report file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"eval_report_{ts}.html"

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = "\n".join([
        _metrics_section(metrics),
        _confusion_section(metrics),
        _bucket_summary_section(buckets, metrics["total"]),
        _fp_fn_section(buckets),
        _contradictory_section(buckets),
        _judge_section(judge_results),
        _confidence_dist_section(df),
    ])

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AD/ADRD Eval Report — {now_str}</title>
<style>{_CSS}</style>
</head>
<body>
<section>
  <h1>AD/ADRD CyberDoctor — Evaluation Report</h1>
  <p class="header-meta">Generated: {now_str} &nbsp;|&nbsp;
     Accuracy: <strong>{metrics['accuracy']:.4f}</strong> &nbsp;|&nbsp;
     Sensitivity: <strong>{metrics['sensitivity']:.4f}</strong> &nbsp;|&nbsp;
     PPV: <strong>{metrics['ppv']:.4f}</strong>
  </p>
</section>
{body}
</body>
</html>"""

    report_path.write_text(html_doc, encoding="utf-8")
    return report_path
