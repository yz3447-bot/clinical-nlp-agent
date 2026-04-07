# AD/ADRD CyberDoctor — Horizontal Multi-Agent System

**Detects Alzheimer's Disease and related dementias (AD/ADRD) from hospital EHR discharge summaries using three parallel AI specialist agents, eliminating the false positives caused by naive keyword matching or ICD-code lookups alone.**

---

## Results

| Dataset | Records | Labeled | AD/ADRD Positive | Avg Note Length |
|---|---|---|---|---|
| MIMIC-style EHR discharge summaries | 100 | 87 (excl. uncertain) | ~54% | ~11,200 chars |

Metrics (accuracy, sensitivity, PPV) are printed automatically after each run. Reproduce with:
```bash
python main_c.py
```

---

## Architecture

```
EHR Record (full text + ICD codes)
         │
         ├────────────────────┬────────────────────┐
         ▼                    ▼                    ▼
 ┌───────────────┐  ┌──────────────────┐  ┌────────────────┐
 │ ClinTextAgent │  │ MedicationAgent  │  │ DiagnosisAgent │
 │  neurologist  │  │   pharmacist     │  │ coding spec.   │
 │   weight: 3   │  │   weight: 3      │  │   weight: 1    │
 └───────┬───────┘  └────────┬─────────┘  └───────┬────────┘
         │                   │                    │
         └───────────────────┴────────────────────┘
                             │  (all three run in parallel)
                             ▼
                  ┌─────────────────────┐
                  │  SynthesizerAgent   │
                  │ attending physician │
                  │ weighs all reports, │
                  │ resolves conflicts, │
                  │ can overrule agents │
                  └──────────┬──────────┘
                             ▼
              final_prediction (0/1) · subtype · confidence
              causal_chain · contributing_agents · discrepancy
```

| Agent | Role | Method |
|---|---|---|
| **ClinTextAgent** | Identifies current cognitive symptoms | LLM reads full note as neurologist |
| **MedicationAgent** | Checks for active AD prescriptions | LLM reads full note as pharmacist |
| **DiagnosisAgent** | Validates ICD codes as current vs. historical | Rule-based match → LLM context check |
| **SynthesizerAgent** | Final diagnosis + subtype | LLM with full evidence; rule-based fallback |

---

## Quickstart

**Prerequisites:** Python 3.10+, a Google Gemini API key, and your EHR data CSV in `data/`.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export GEMINI_API_KEY=your_key_here   # Windows: set GEMINI_API_KEY=your_key_here

# 3. Run the full pipeline
python main_c.py

# 4. (Optional) Test on 5 records with verbose output
python test_run.py

# 5. (Optional) Generate interactive HTML report
python visualize.py
# → opens outputs/report.html
```

**REST API (FastAPI):**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000

# Single-record inference
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with progressive memory loss...", "icd_codes": "F0280|G309"}'
```

API docs at `http://localhost:8000/docs`.

**Expected output per record:**
```json
{
  "prediction": 1,
  "subtype": "ad",
  "confidence": "high",
  "causal_chain": "ClinTextAgent found explicit dementia documentation...",
  "contributing_agents": ["ClinTextAgent", "MedicationAgent"],
  "discrepancy": null,
  "latency_ms": 4821.3
}
```

---

## Engineering Design Decisions

### 1. Parallel Agent Execution
**What:** All three specialist agents run simultaneously using `ThreadPoolExecutor(max_workers=3)`.

**Why:** Each agent makes an independent LLM call (~2–5s each). Running them sequentially would triple the latency with no benefit — they share no state.

**Impact:** ~3× faster per record than a sequential pipeline. At 4 LLM calls per record, total latency is bounded by the slowest agent rather than their sum.

---

### 2. Weighted Evidence Fusion
**What:** MedicationAgent and ClinTextAgent carry weight 3; DiagnosisAgent carries weight 1. These weights inform the Synthesizer's scoring, not a hard vote.

**Why:** A current AD medication prescription means a physician has *already diagnosed* the patient and started treatment — that is the strongest possible indirect confirmation. ICD codes alone are weakest: they are routinely carried forward from previous admissions without reflecting the current encounter.

**Impact:** The system mirrors real clinical reasoning. Medication evidence triggers high-confidence positive predictions; ICD codes alone trigger low-confidence ones that the Synthesizer can override.

---

### 3. Contradiction Detection and Auditability
**What:** The Synthesizer explicitly detects disagreements between agents, records which agents were overruled, and writes the full reasoning chain to `causal_chain` and `overruled_agents`.

**Why:** Medical AI must be auditable. A black-box "prediction: 1" is not actionable for a clinician. Every decision needs to explain *why* — especially when agents contradict each other (e.g., ICD codes present but note shows an orthopedic admission with dementia only in past history).

**Impact:** Full decision traceability. Reviewers can inspect exactly which agent was trusted, which was overruled, and why — for every single prediction.

---

### 4. Graceful Fallback — Zero Silent Failures
**What:** If the Synthesizer's LLM call fails or returns unparseable output, the system automatically falls back to a deterministic rule-based decision (medication → ICD codes → symptoms, in priority order).

**Why:** Silent failures are the worst outcome in a medical system — returning a wrong confident answer with no indication something went wrong. The fallback trades LLM reasoning for a safe, explainable rule that always produces a valid output.

**Impact:** The pipeline never crashes mid-run or returns a null prediction. The fallback result is clearly labeled `[Fallback — LLM failed]` in `causal_chain` so downstream reviewers know which records to re-examine.

---

### 5. Cost-Optimized ICD Lookup
**What:** DiagnosisAgent runs a rule-based ICD whitelist match *before* making any LLM call. If no AD/ADRD codes are present in the record, it returns immediately with no API cost.

**Why:** ~40% of records have no relevant ICD codes. Sending those to an LLM just to confirm "nothing found" wastes API quota and adds latency.

**Impact:** Roughly 40% reduction in LLM calls for the DiagnosisAgent. The LLM is only invoked when there is actual evidence to evaluate (current vs. historical coding decision).

---

### 6. Single LLM Initialization via Lifespan Pattern
**What:** In the FastAPI service (`api.py`), the Gemini LLM client is initialized once at application startup using FastAPI's `lifespan` context manager — not recreated on every request.

**Why:** Creating an LLM client involves authentication, SDK setup, and connection overhead. Rebuilding it per request adds ~100–300ms of unnecessary latency and wastes resources under concurrent load.

**Impact:** The LLM client is shared across all requests. Under load, this avoids resource contention and keeps per-request latency predictable.

---

## File Structure

```
ad_cyberdoctor_horizontal/
├── main_c.py               # Batch pipeline: load CSV, run agents, write results, print metrics
├── api.py                  # FastAPI service: POST /classify, GET /health
├── requirements.txt        # Python dependencies
├── visualize.py            # Generates outputs/report.html (interactive charts)
├── agents/
│   ├── clintext_agent.py   # Neurologist: symptom evidence from full note
│   ├── medication_agent.py # Pharmacist: current AD medication detection
│   ├── diagnosis_agent.py  # Coding specialist: ICD match + current/historical check
│   └── synthesizer_agent.py# Attending physician: final diagnosis + subtype
├── data/
│   └── data_test.csv       # Input EHR data (place your CSV here)
└── outputs/
    ├── predictions_c.csv   # Per-record predictions with full reasoning
    └── report.html         # Interactive visualization report
```

---

## Output Schema

Each record in `outputs/predictions_c.csv`:

| Column | Description |
|---|---|
| `final_prediction` | **1** = AD/ADRD present, **0** = not present |
| `subtype` | `ad` / `vd` / `ftd` / `nsd` / `na` |
| `confidence` | `high` / `medium` / `low` |
| `causal_chain` | Synthesizer's step-by-step reasoning |
| `contributing_agents` | Agents with positive findings |
| `discrepancy` | Conflicts between agents and how resolved |
| `overruled_agents` | Agents the Synthesizer disagreed with |
| `dx_found` / `meds_found` / `symptoms_found` | Per-agent signal flags |

Processing supports **checkpoint/resume**: records already in `predictions_c.csv` are skipped on restart.

---

## LLM Configuration

| Parameter | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| Temperature | `0` (deterministic) |
| Context window | 1,000,000 tokens (full notes, no truncation) |
| LLM calls per record | Up to 4 (3 agents + synthesizer) |
| Rate limit buffer | 30s between records (batch mode) |
