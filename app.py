"""
app.py — Streamlit frontend for the AD/ADRD CyberDoctor system.

Pages:
  1. Classify     — single-record inference via the FastAPI backend
  2. Review Queue — approve / reject cases (bulk import from predictions_c.csv)
  3. Eval Dashboard — metrics, confusion matrix, error analysis

Run:
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ─── Configuration ────────────────────────────────────────────────────────────

# Override with API_BASE env var when running inside Docker
# (docker-compose sets API_BASE=http://api:8000 for the streamlit service)
API_BASE   = os.environ.get("API_BASE", "http://localhost:8000")
_DATA_PATH = Path(__file__).parent / "data"    / "data_test.csv"
_PRED_PATH = Path(__file__).parent / "outputs" / "predictions_c.csv"
_GT_COL    = "is AD/ADRD? (type in 1,0, -1) 1 for yes ADRD Present, 0 for No ADRD, -1 uncertain)"

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AD/ADRD CyberDoctor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar navigation ───────────────────────────────────────────────────────

st.sidebar.title("AD/ADRD CyberDoctor")
st.sidebar.caption("Horizontal Multi-Agent System")
page = st.sidebar.radio(
    "Navigation",
    ["Classify", "Review Queue", "Eval Dashboard"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(f"API: `{API_BASE}`")


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _prediction_label(pred: int) -> str:
    return "AD/ADRD Present" if pred == 1 else "No AD/ADRD"


def _call_classify(text: str, icd_codes: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/classify",
            json={"text": text, "icd_codes": icd_codes},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to API at {API_BASE}. "
            "Make sure the FastAPI server is running:  \n"
            "`uvicorn api:app --host 0.0.0.0 --port 8000`"
        )
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


def _load_dataset() -> pd.DataFrame | None:
    if not _DATA_PATH.exists():
        st.warning(f"Dataset not found: {_DATA_PATH}")
        return None
    return pd.read_csv(_DATA_PATH, dtype=str).fillna("")


def _add_to_review(note_id: str, text: str, result: dict) -> None:
    from review_queue import add_to_queue
    add_to_queue(
        note_id=note_id,
        text=text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        confidence_score_clin=result.get("confidence_score_clin", 0.0),
        confidence_score_med=result.get("confidence_score_med", 0.0),
        confidence_score_dx=result.get("confidence_score_dx", 0.0),
        causal_chain=result.get("causal_chain", ""),
        discrepancy=result.get("discrepancy") or "",
    )


def _esc(s: str) -> str:
    """HTML-escape a string for safe inline injection."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ─── Classify result HTML renderer ───────────────────────────────────────────

_CLASSIFY_CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:transparent;color:#e0e0e0;padding:4px 0}
/* ── Agent grid ── */
.agents-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}
.agent-card{background:#16213e;border:1px solid #2a3a6e;border-radius:12px;overflow:hidden}
.agent-card-top{padding:16px 16px 12px;cursor:pointer;user-select:none}
.agent-header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:10px}
.agent-left{}
.agent-name{font-size:.88rem;font-weight:700;color:#e0e0e0;line-height:1.2}
.agent-role{font-size:.73rem;color:#8892b0;margin-top:2px}
.agent-type-tag{padding:2px 9px;border-radius:10px;font-size:.7rem;font-weight:600;white-space:nowrap;margin-top:4px;display:inline-block}
.type-llm   {background:#3d1a6e;color:#c084fc}
.type-hybrid{background:#4a2a00;color:#fb923c}
.type-rule  {background:#0a2a4a;color:#60a5fa}
.found-icon{font-size:1.8rem;font-weight:900;line-height:1}
.found-yes{color:#22c55e}
.found-no {color:#ef4444}
/* ── Progress bar ── */
.bar-wrap{margin:6px 0 4px}
.bar-label{font-size:.72rem;color:#8892b0;display:flex;justify-content:space-between}
.bar-bg{background:#0d1b30;border-radius:6px;height:7px;margin-top:4px}
.bar-fill{background:linear-gradient(90deg,#3f51b5,#7c4dff);border-radius:6px;height:7px}
/* ── Toggle ── */
.toggle-btn{width:100%;padding:7px;background:#0d1b30;border:none;border-top:1px solid #2a3a6e;
            color:#60a5fa;font-size:.76rem;cursor:pointer;
            display:flex;align-items:center;justify-content:center;gap:4px}
.toggle-arrow{display:inline-block;transition:transform .2s;font-size:.68rem}
.card-detail{display:none;padding:12px 16px 14px;border-top:1px solid #2a3a6e}
.card-detail.open{display:block}
/* ── Quote boxes ── */
.quote-box{background:#0d1b30;border-left:3px solid #3f51b5;border-radius:4px;
           padding:7px 11px;font-size:.79rem;color:#b0bec5;margin-bottom:7px;
           line-height:1.5;word-break:break-word}
.detail-label{font-size:.71rem;font-weight:700;color:#8892b0;text-transform:uppercase;
              letter-spacing:.5px;margin-bottom:5px;margin-top:8px}
.detail-label:first-child{margin-top:0}
.pill{display:inline-block;padding:2px 9px;border-radius:10px;font-size:.74rem;
      font-weight:600;margin:2px 3px 2px 0}
.pill-green{background:#1a3a1a;color:#4ade80}
.pill-blue {background:#0a2a4a;color:#60a5fa}
.pill-red  {background:#3a0a0a;color:#f87171}
.pill-grey {background:#1e2a3a;color:#94a3b8}
/* ── Verdict section ── */
.verdict-section{background:#16213e;border:1px solid #2a3a6e;border-radius:12px;
                 padding:20px 22px;margin-bottom:16px}
.verdict-main{font-size:2rem;font-weight:800;margin-bottom:12px;letter-spacing:-.5px}
.verdict-pos{color:#22c55e}
.verdict-neg{color:#ef4444}
.verdict-tags{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;align-items:center}
.tag{display:inline-block;padding:3px 11px;border-radius:10px;font-size:.78rem;font-weight:600}
.tag-high  {background:#1a3a1a;color:#4ade80}
.tag-medium{background:#3a2a00;color:#fbbf24}
.tag-low   {background:#3a0a0a;color:#f87171}
.tag-mode  {background:#1e2a3a;color:#94a3b8}
/* ── Warning / alert boxes ── */
.warn-box{background:#2a1f00;border-left:3px solid #f59e0b;border-radius:6px;
          padding:10px 14px;font-size:.82rem;color:#fcd34d;margin:10px 0}
.error-box{background:#2a0a0a;border-left:3px solid #ef4444;border-radius:6px;
           padding:10px 14px;font-size:.82rem;color:#fca5a5;margin:10px 0}
.reason-box{background:#0d1b30;border-left:3px solid #8892b0;border-radius:6px;
            padding:12px 14px;font-size:.82rem;color:#b0bec5;line-height:1.65;
            margin:10px 0;white-space:pre-wrap;word-break:break-word;max-height:260px;overflow-y:auto}
.section-label{font-size:.78rem;font-weight:700;color:#8892b0;text-transform:uppercase;
               letter-spacing:.5px;margin:14px 0 6px}
/* ── Metrics row ── */
.metrics-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}
.metric-box{background:#16213e;border:1px solid #2a3a6e;border-radius:10px;
            padding:14px 10px;text-align:center}
.metric-val{font-size:1.45rem;font-weight:700;color:#60a5fa}
.metric-lbl{font-size:.74rem;color:#8892b0;margin-top:4px}
.overruled-chip{display:inline-block;padding:2px 9px;background:#3a0a0a;color:#f87171;
                border-radius:10px;font-size:.75rem;font-weight:600;margin:2px 3px 2px 0}
</style>
"""

_CLASSIFY_JS = """
<script>
function toggleDetail(id) {
  var detail = document.getElementById('detail-' + id);
  var arrow  = document.getElementById('arrow-' + id);
  var btn    = document.getElementById('btn-' + id);
  if (!detail) return;
  if (detail.classList.contains('open')) {
    detail.classList.remove('open');
    if (arrow) arrow.style.transform = 'rotate(0deg)';
    if (btn)   btn.childNodes[1].nodeValue = ' Show details';
  } else {
    detail.classList.add('open');
    if (arrow) arrow.style.transform = 'rotate(180deg)';
    if (btn)   btn.childNodes[1].nodeValue = ' Hide details';
  }
}
</script>
"""


def _bar(score: float, label: str) -> str:
    pct = round(score * 100)
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-label"><span>{label}</span><span>{score:.2f}</span></div>'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%"></div></div>'
        f'</div>'
    )


def _agent_card_clin(result: dict, idx: str) -> str:
    findings = result.get("agent_findings") or {}
    found    = findings.get("symptoms_found", "ClinTextAgent" in result.get("contributing_agents", []))
    score    = result.get("confidence_score_clin", 0.0)
    evidence = findings.get("evidence_list", [])
    reasoning = _esc(findings.get("clin_reasoning", ""))

    icon_cls = "found-yes" if found else "found-no"
    icon_chr = "✓" if found else "✗"

    quotes_html = ""
    for ev in evidence[:3]:
        quotes_html += f'<div class="quote-box">{_esc(str(ev))}</div>'
    if not quotes_html:
        quotes_html = '<div class="quote-box" style="color:#5a6a7a">No direct quotes extracted.</div>'

    return f"""
<div class="agent-card">
  <div class="agent-card-top" onclick="toggleDetail('{idx}')">
    <div class="agent-header">
      <div class="agent-left">
        <div class="agent-name">ClinTextAgent</div>
        <div class="agent-role">L1 &middot; Clinical Text</div>
        <span class="agent-type-tag type-llm">LLM-based</span>
      </div>
      <span class="found-icon {icon_cls}">{icon_chr}</span>
    </div>
    {_bar(score, "Confidence Score")}
  </div>
  <button class="toggle-btn" id="btn-{idx}" onclick="toggleDetail('{idx}')">
    <span class="toggle-arrow" id="arrow-{idx}" style="transform:rotate(0deg)">&#9660;</span>
    &nbsp;Show details
  </button>
  <div class="card-detail" id="detail-{idx}">
    <div class="detail-label">Evidence Quotes</div>
    {quotes_html}
    {"<div class='detail-label'>Reasoning</div><div class='quote-box'>" + reasoning + "</div>" if reasoning else ""}
  </div>
</div>"""


def _agent_card_med(result: dict, idx: str) -> str:
    findings = result.get("agent_findings") or {}
    found    = findings.get("meds_found", "MedicationAgent" in result.get("contributing_agents", []))
    score    = result.get("confidence_score_med", 0.0)
    meds     = findings.get("medications", [])
    status   = findings.get("med_status", "none")
    source   = findings.get("med_source", "text")

    icon_cls = "found-yes" if found else "found-no"
    icon_chr = "✓" if found else "✗"

    status_cls = "pill-green" if status == "current" else ("pill-grey" if status in ("historical", "refused") else "pill-red")
    source_cls = "pill-blue" if source == "structured" else "pill-grey"

    meds_html = ""
    if meds:
        meds_html = '<div class="detail-label">Medications</div>'
        for m in meds:
            meds_html += f'<span class="pill pill-green">{_esc(str(m))}</span>'
    else:
        meds_html = '<div class="quote-box" style="color:#5a6a7a">No AD medications detected.</div>'

    return f"""
<div class="agent-card">
  <div class="agent-card-top" onclick="toggleDetail('{idx}')">
    <div class="agent-header">
      <div class="agent-left">
        <div class="agent-name">MedicationAgent</div>
        <div class="agent-role">L2 &middot; Medications</div>
        <span class="agent-type-tag type-hybrid">Hybrid</span>
      </div>
      <span class="found-icon {icon_cls}">{icon_chr}</span>
    </div>
    {_bar(score, "Confidence Score")}
  </div>
  <button class="toggle-btn" id="btn-{idx}" onclick="toggleDetail('{idx}')">
    <span class="toggle-arrow" id="arrow-{idx}" style="transform:rotate(0deg)">&#9660;</span>
    &nbsp;Show details
  </button>
  <div class="card-detail" id="detail-{idx}">
    {meds_html}
    <div class="detail-label">Status &amp; Source</div>
    <span class="pill {status_cls}">{_esc(status)}</span>
    <span class="pill {source_cls}">source: {_esc(source)}</span>
  </div>
</div>"""


def _agent_card_dx(result: dict, idx: str) -> str:
    findings   = result.get("agent_findings") or {}
    found      = findings.get("dx_found", "DiagnosisAgent" in result.get("contributing_agents", []))
    score      = result.get("confidence_score_dx", 0.0)
    codes      = findings.get("matched_codes", [])
    is_current = findings.get("is_current_diagnosis", False)
    dx_reason  = _esc(findings.get("dx_reasoning", ""))

    icon_cls = "found-yes" if found else "found-no"
    icon_chr = "✓" if found else "✗"

    codes_html = ""
    if codes:
        codes_html = '<div class="detail-label">Matched ICD Codes</div>'
        for c in codes:
            codes_html += f'<span class="pill pill-blue">{_esc(str(c))}</span>'
    else:
        codes_html = '<div class="quote-box" style="color:#5a6a7a">No AD/ADRD ICD codes matched.</div>'

    current_cls = "pill-green" if is_current else "pill-grey"
    current_lbl = "Current diagnosis" if is_current else "Historical / not current"

    return f"""
<div class="agent-card">
  <div class="agent-card-top" onclick="toggleDetail('{idx}')">
    <div class="agent-header">
      <div class="agent-left">
        <div class="agent-name">DiagnosisAgent</div>
        <div class="agent-role">L3 &middot; Diagnosis ICD</div>
        <span class="agent-type-tag type-rule">Rule-based</span>
      </div>
      <span class="found-icon {icon_cls}">{icon_chr}</span>
    </div>
    {_bar(score, "Confidence Score")}
  </div>
  <button class="toggle-btn" id="btn-{idx}" onclick="toggleDetail('{idx}')">
    <span class="toggle-arrow" id="arrow-{idx}" style="transform:rotate(0deg)">&#9660;</span>
    &nbsp;Show details
  </button>
  <div class="card-detail" id="detail-{idx}">
    {codes_html}
    <div class="detail-label">Currency Assessment</div>
    <span class="pill {current_cls}">{current_lbl}</span>
    {"<div class='detail-label'>Reasoning</div><div class='quote-box'>" + dx_reason + "</div>" if dx_reason else ""}
  </div>
</div>"""


def _verdict_html(result: dict) -> str:
    pred        = result["prediction"]
    conf        = result.get("confidence", "low")
    subtype     = result.get("subtype", "na").upper()
    mode        = result.get("synthesis_mode", "").replace("_", " ").title()
    causal      = _esc(result.get("causal_chain", ""))
    discrepancy = result.get("discrepancy") or ""
    correction  = result.get("confidence_correction") or ""
    overruled   = result.get("overruled_agents", []) or []
    if isinstance(overruled, str):
        try:
            import json as _json
            overruled = _json.loads(overruled.replace("'", '"'))
        except Exception:
            overruled = [overruled] if overruled else []

    latency  = result.get("latency_ms", 0)
    calls    = result.get("llm_calls", "—")
    cost_usd = result.get("estimated_cost_usd", 0)

    verdict_cls = "verdict-pos" if pred == 1 else "verdict-neg"
    verdict_txt = "AD/ADRD Present" if pred == 1 else "No AD / ADRD"

    conf_tag_cls = {"high": "tag-high", "medium": "tag-medium", "low": "tag-low"}.get(conf, "tag-mode")

    overruled_html = ""
    if overruled:
        chips = "".join(f'<span class="overruled-chip">{_esc(str(a))}</span>' for a in overruled)
        overruled_html = f'<div class="section-label">Overruled Agents</div><div>{chips}</div>'

    disc_html = ""
    if discrepancy:
        disc_html = f'<div class="error-box">&#9888; Agent Discrepancy: {_esc(discrepancy)}</div>'

    warn_html = ""
    if correction:
        warn_html = f'<div class="warn-box">&#9888; Confidence correction: {_esc(correction)}</div>'

    metrics_html = f"""
<div class="metrics-row">
  <div class="metric-box"><div class="metric-val">{round(latency)}&thinsp;ms</div><div class="metric-lbl">Latency</div></div>
  <div class="metric-box"><div class="metric-val">{calls}</div><div class="metric-lbl">LLM Calls</div></div>
  <div class="metric-box"><div class="metric-val">${cost_usd:.4f}</div><div class="metric-lbl">Est. Cost</div></div>
</div>"""

    return f"""
<div class="verdict-section">
  <div class="verdict-main {verdict_cls}">{verdict_txt}</div>
  <div class="verdict-tags">
    <span class="tag {conf_tag_cls}">{conf.upper()}</span>
    <span class="tag tag-mode">Subtype: {_esc(subtype)}</span>
    <span class="tag tag-mode">{_esc(mode)}</span>
  </div>
  {warn_html}
  {disc_html}
  {overruled_html}
  <div class="section-label">Reasoning Chain</div>
  <div class="reason-box">{causal}</div>
  {metrics_html}
</div>"""


def _render_result(result: dict, note_id: str = "", text: str = "") -> None:
    """Render the full classification result using report.html-style HTML cards."""

    # Always add every classify result to the review queue
    try:
        _add_to_review(note_id or "manual", text, result)
    except Exception:
        pass

    cards_html = (
        '<div class="agents-grid">'
        + _agent_card_clin(result, "clin")
        + _agent_card_med(result, "med")
        + _agent_card_dx(result, "dx")
        + '</div>'
    )
    verdict = _verdict_html(result)

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">{_CLASSIFY_CSS}</head>
<body>
{cards_html}
{verdict}
{_CLASSIFY_JS}
</body></html>"""

    # Estimate height: agents (~220px) + verdict (~420px base + extras)
    extra = 60 * (1 if result.get("discrepancy") else 0)
    extra += 50 * (1 if result.get("confidence_correction") else 0)
    components.html(full_html, height=700 + extra, scrolling=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════

if page == "Classify":
    st.title("Classify Clinical Note")
    st.caption("Run the 3-agent pipeline against the FastAPI backend.")

    mode = st.radio("Input mode", ["Dataset record", "Manual input"], horizontal=True)

    if mode == "Dataset record":
        df = _load_dataset()
        if df is not None:
            labeled      = df[df[_GT_COL].isin(["0", "1"])].copy()
            selected_id  = st.selectbox("Select note_id", labeled["note_id"].tolist())
            row          = labeled[labeled["note_id"] == selected_id].iloc[0]
            gt_val       = row[_GT_COL]

            st.caption(
                f"Ground truth: **{_prediction_label(int(gt_val))}** | "
                f"ICD codes: `{str(row.get('all_icd_codes', ''))[:80]}...`"
            )
            with st.expander("Clinical note preview (first 500 chars)"):
                st.text(str(row["text"])[:500])

            icd_input = st.text_input(
                "ICD codes (pipe-separated)",
                value=str(row.get("all_icd_codes", "")),
            )

            if st.button("Run Classification", type="primary"):
                with st.spinner("Running multi-agent pipeline..."):
                    result = _call_classify(str(row["text"]), icd_input)
                if result:
                    _render_result(result, note_id=selected_id, text=str(row["text"]))

    else:  # Manual input
        note_text = st.text_area(
            "Paste clinical note text",
            height=250,
            placeholder="Patient presents with progressive memory loss...",
        )
        icd_input = st.text_input(
            "ICD codes (pipe-separated, optional)",
            placeholder="F0280|G309|3310",
        )

        if st.button("Run Classification", type="primary", disabled=not note_text.strip()):
            with st.spinner("Running multi-agent pipeline..."):
                result = _call_classify(note_text.strip(), icd_input.strip())
            if result:
                _render_result(result, note_id="manual", text=note_text.strip())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: REVIEW QUEUE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Review Queue":
    from review_queue import (
        get_all_cases,
        get_pending_cases,
        get_reviewed_cases,
        approve_case,
        reject_case,
        init_db,
        bulk_import_from_csv,
    )

    init_db()

    st.title("Review Queue")

    # ── Counter cards ─────────────────────────────────────────────────────────
    all_cases = get_all_cases()
    n_pending  = sum(1 for c in all_cases if c["status"] == "pending")
    n_approved = sum(1 for c in all_cases if c["status"] == "approved")
    n_rejected = sum(1 for c in all_cases if c["status"] == "rejected")

    cnt1, cnt2, cnt3, cnt4 = st.columns(4)
    cnt1.metric("Total",    len(all_cases))
    cnt2.metric("Pending",  n_pending)
    cnt3.metric("Approved", n_approved)
    cnt4.metric("Rejected", n_rejected)

    # ── Bulk import button ────────────────────────────────────────────────────
    if st.button("Import from predictions_c.csv", type="secondary"):
        if not _PRED_PATH.exists():
            st.warning("predictions_c.csv not found. Run `python main_c.py` first.")
        else:
            imported = bulk_import_from_csv(_PRED_PATH)
            if imported > 0:
                st.success(f"Imported {imported} new case(s) from predictions_c.csv.")
            else:
                st.info("No new cases to import (all note_ids already in queue).")
            st.rerun()

    st.divider()

    # ── Case cards ────────────────────────────────────────────────────────────
    all_cases = get_all_cases()  # refresh after possible import

    if not all_cases:
        st.info(
            "Queue is empty. Run a classification or click 'Import from predictions_c.csv' "
            "to populate it."
        )
    else:
        # Filter controls
        filter_status = st.radio(
            "Show",
            ["All", "Pending", "Approved", "Rejected"],
            horizontal=True,
            label_visibility="collapsed",
        )

        show_cases = [
            c for c in all_cases
            if filter_status == "All" or c["status"] == filter_status.lower()
        ]

        st.caption(f"Showing {len(show_cases)} case(s)")

        for case in show_cases:
            status   = case["status"]
            pred_lbl = _prediction_label(case["prediction"])
            pred_col = "green" if case["prediction"] == 1 else "red"
            conf     = case.get("confidence", "low")
            conf_col = {"high": "green", "medium": "orange", "low": "red"}.get(conf, "grey")

            header = (
                f"**#{case['id']}** — `{case['note_id']}` | "
                f":{pred_col}[{pred_lbl}] | "
                f":{conf_col}[{conf.upper()}] | "
                f"_{status.upper()}_ | {case.get('created_at', '')}"
            )

            with st.expander(header, expanded=(status == "pending")):
                if case.get("text_snippet"):
                    st.caption(f"Note snippet: {case['text_snippet']}")

                # Confidence score bars (HTML)
                cs_clin = float(case.get("confidence_score_clin") or 0)
                cs_med  = float(case.get("confidence_score_med")  or 0)
                cs_dx   = float(case.get("confidence_score_dx")   or 0)

                bar_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
{_CLASSIFY_CSS}</head><body>
<div style="padding:4px 0">
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
    <div>{_bar(cs_clin, "ClinTextAgent")}</div>
    <div>{_bar(cs_med,  "MedicationAgent")}</div>
    <div>{_bar(cs_dx,   "DiagnosisAgent")}</div>
  </div>
</div>
</body></html>"""
                components.html(bar_html, height=60)

                # Causal chain (collapsible)
                if case.get("causal_chain"):
                    with st.expander("Reasoning chain"):
                        st.text(case["causal_chain"])

                # Discrepancy
                if case.get("discrepancy"):
                    st.error(f"Agent Discrepancy: {case['discrepancy']}")

                # Actions
                if status == "pending":
                    col_a, col_r = st.columns(2)
                    approve_key = f"approve_{case['id']}"
                    reject_key  = f"reject_{case['id']}"
                    comment_key = f"comment_{case['id']}"

                    comment = st.text_input("Reviewer comment (optional)", key=comment_key)

                    with col_a:
                        if st.button("✓ Approve", key=approve_key, type="primary", use_container_width=True):
                            approve_case(case["id"], comment)
                            st.success("Approved.")
                            st.rerun()
                    with col_r:
                        if st.button("✗ Reject", key=reject_key, use_container_width=True):
                            reject_case(case["id"], comment)
                            st.warning("Rejected.")
                            st.rerun()

                elif status == "approved":
                    st.success(f"✓ Approved" + (f" — {case['reviewer_comment']}" if case.get("reviewer_comment") else ""))

                elif status == "rejected":
                    st.error(f"✗ Rejected" + (f" — {case['reviewer_comment']}" if case.get("reviewer_comment") else ""))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EVAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Eval Dashboard":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("Eval Dashboard")
    st.caption(f"Based on: `{_PRED_PATH}`")

    if not _PRED_PATH.exists():
        st.warning(
            "No predictions file found. "
            "Run `python main_c.py` first to generate `outputs/predictions_c.csv`."
        )
        st.stop()

    # ── Load predictions ──────────────────────────────────────────────────────
    df_raw = pd.read_csv(_PRED_PATH, dtype=str).fillna("")
    df = df_raw[df_raw["ground_truth"].isin(["0", "1"])].copy()
    if df.empty:
        st.error("No labeled rows in predictions_c.csv (ground_truth must be 0 or 1).")
        st.stop()

    df["ground_truth"]     = df["ground_truth"].astype(int)
    df["final_prediction"] = df["final_prediction"].astype(int)

    for col, default in {
        "synthesis_mode":        "unknown",
        "confidence_score_clin": 0.0,
        "confidence_score_med":  0.0,
        "confidence_score_dx":   0.0,
        "confidence_correction": "",
    }.items():
        if col not in df.columns:
            df[col] = default

    for col in ("confidence_score_clin", "confidence_score_med", "confidence_score_dx"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

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
    f1 = (2 * ppv * sensitivity / (ppv + sensitivity)
          if (ppv + sensitivity) > 0 else 0.0)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.subheader("Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",    f"{accuracy:.1%}")
    m2.metric("Sensitivity", f"{sensitivity:.1%}")
    m3.metric("PPV",         f"{ppv:.1%}")
    m4.metric("Specificity", f"{specificity:.1%}")
    m5.metric("F1",          f"{f1:.1%}")

    st.caption(f"Labeled records: {total}  |  Correct: {correct}  |  Errors: {total - correct}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["Actual Positive", "Actual Negative"],
        columns=["Predicted Positive", "Predicted Negative"],
    )

    def _cm_style(val):
        if val == tp or val == tn:
            return "background-color: #1a472a; color: #a8d5b5"
        return "background-color: #5c1a1a; color: #f4a0a0"

    st.dataframe(cm_df.style.applymap(_cm_style), use_container_width=False)

    # ── Error bucket bar chart ────────────────────────────────────────────────
    st.subheader("Error Buckets")
    errors_df  = df[df["ground_truth"] != df["final_prediction"]]
    has_disc   = df["discrepancy"].str.strip() != ""
    is_low_conf = df["confidence"] == "low"

    bucket_data = {
        "False Positives":       len(errors_df[errors_df["final_prediction"] == 1]),
        "False Negatives":       len(errors_df[errors_df["final_prediction"] == 0]),
        "Contradictory errors":  int(has_disc[errors_df.index].sum()),
        "Low-confidence errors": int(is_low_conf[errors_df.index].sum()),
        "All w/ discrepancy":    int(has_disc.sum()),
    }

    fig_buckets = go.Figure(go.Bar(
        x=list(bucket_data.keys()),
        y=list(bucket_data.values()),
        marker_color=["#e05252", "#e09552", "#e0d452", "#52a8e0", "#7b52e0"],
        text=list(bucket_data.values()),
        textposition="outside",
    ))
    fig_buckets.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        margin=dict(t=30, b=10),
        height=320,
        yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig_buckets, use_container_width=True)

    # ── Confidence score histograms ───────────────────────────────────────────
    score_cols = ["confidence_score_clin", "confidence_score_med", "confidence_score_dx"]
    if df[score_cols].sum().sum() > 0:
        st.subheader("Agent Confidence Score Distributions")
        hcol1, hcol2, hcol3 = st.columns(3)
        names  = {"confidence_score_clin": "ClinTextAgent",
                  "confidence_score_med":  "MedicationAgent",
                  "confidence_score_dx":   "DiagnosisAgent"}
        colors = {"confidence_score_clin": "#4a9eff",
                  "confidence_score_med":  "#52e0a8",
                  "confidence_score_dx":   "#e052a8"}

        for col, hcol in zip(score_cols, [hcol1, hcol2, hcol3]):
            vals = df[col][df[col] > 0]
            if vals.empty:
                hcol.caption(f"{names[col]}: all zero")
                continue
            fig_h = px.histogram(
                vals, nbins=10,
                color_discrete_sequence=[colors[col]],
                labels={"value": "Score", "count": "Cases"},
            )
            fig_h.update_layout(
                title=names[col],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10),
                height=220,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(gridcolor="#333"),
            )
            hcol.plotly_chart(fig_h, use_container_width=True)

    # ── Contradictory cases table ─────────────────────────────────────────────
    contradictory = df[has_disc].copy()
    if not contradictory.empty:
        st.subheader(f"Contradictory Cases ({len(contradictory)})")
        st.caption("Cases where agent findings disagreed — resolved by SynthesizerAgent.")

        display_cols = [c for c in ["note_id", "ground_truth", "final_prediction",
                                     "confidence", "discrepancy"] if c in contradictory.columns]
        show_df = contradictory[display_cols].copy()
        show_df["ground_truth"]     = show_df["ground_truth"].map({1: "Positive", 0: "Negative"})
        show_df["final_prediction"] = show_df["final_prediction"].map({1: "Positive", 0: "Negative"})

        def _highlight_errors(row):
            if row.get("ground_truth") != row.get("final_prediction"):
                return ["background-color: #3a1a1a"] * len(row)
            return [""] * len(row)

        st.dataframe(
            show_df.style.apply(_highlight_errors, axis=1),
            use_container_width=True,
            hide_index=True,
        )
