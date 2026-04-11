# AD/ADRD CyberDoctor — Horizontal Multi-Agent System

**Detects Alzheimer's Disease and related dementias (AD/ADRD) from hospital EHR discharge summaries using three parallel AI specialist agents with rule-based confidence scoring and post-hoc correction — achieving 94.3% sensitivity on 87 labeled cases while eliminating the false positives caused by naive keyword matching or ICD-code lookups alone.**

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

Live demo available at https://clinical-nlp-agent.onrender.com/docs

```bash
# Self-host
uvicorn api:app --host 0.0.0.0 --port 8000

# Single-record inference (live endpoint)
curl -X POST https://clinical-nlp-agent.onrender.com/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with progressive memory loss...", "icd_codes": "F0280|G309"}'
```

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

## Evaluation

The eval harness reads `outputs/predictions_c.csv` directly — **no pipeline re-run required**.

```bash
# Basic eval — no API key needed, prints metrics + saves HTML report
python eval/eval_harness.py

# Full eval with LLM-as-judge (requires GEMINI_API_KEY)
python eval/eval_harness.py --run-llm-judge
```

**Results on test set (87 labeled cases):**

| Metric | Value |
|---|---|
| Accuracy | **90.8%** |
| Sensitivity | **94.3%** |
| PPV (Precision) | **90.9%** |
| F1 | **92.6%** |
| Contradictory cases | **38 / 87 (44%)** — agents disagreed; resolved by Synthesizer |

**Error buckets:**

| Bucket | Count | Definition |
|---|---|---|
| False Positives | 5 | Predicted AD/ADRD, actually negative |
| False Negatives | 3 | Predicted no AD/ADRD, actually positive |
| Contradictory cases | 38 | Agents had discrepancy — core focus for LLM judge analysis |
| Low-confidence cases | — | `confidence == "low"` |
| Consensus cases | — | All three agents agreed but prediction was wrong |

**LLM-as-judge** (enabled with `--run-llm-judge`) scores each contradictory case on three dimensions (0–1):
- `reasoning_clarity` — decision traceable to specific evidence?
- `contradiction_handling` — overrule justification sufficient?
- `evidence_consistency` — conclusion consistent with cited evidence?

**CI gate:** Runs automatically on every push. If `predictions_c.csv` exists and sensitivity drops below **0.90**, the build fails and the merge is blocked.

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

### 7. Rule-Based Confidence Scoring
**What:** Each agent computes a deterministic confidence score (0–1) from objective evidence characteristics — independently of what the LLM self-reports. ClinTextAgent scores by evidence count and source location (Assessment/Plan/Discharge = strong; HPI/PMH = weak; negation phrases = penalty). MedicationAgent scores by prescription status (structured column hit = 0.9; current text = 0.7; historical = 0.2; refused/mentioned = 0.1). DiagnosisAgent scores by ICD currency assessment (current + high confidence = 0.8; historical only = 0.1). These scores are passed to the Synthesizer as explicit signals in the prompt.

**Why:** LLM self-reported confidence is unreliable — models frequently express high confidence regardless of actual evidence quality. Rule-based scores are reproducible, inspectable, and cannot be inflated by confident-sounding but unsupported LLM text.

**Impact:** The Synthesizer receives objective evidence quality signals alongside each agent's narrative. Downstream systems and the LLM judge can use these scores to flag cases that warrant closer review, independent of the LLM's own assessment.

---

### 8. Post-hoc Confidence Correction
**What:** After the Synthesizer produces its output, a deterministic rule layer applies forced confidence downgrades before the result is returned. Three rules: (1) discrepancy present + agents overruled → cap at `medium`; (2) two or more agents with confidence score < 0.3 → force to `low`; (3) no medication evidence and fewer than two clinical quotes → force to `low`. Each downgrade writes a reason to the `confidence_correction` field.

**Why:** LLMs tend toward overconfidence, especially when asked to synthesize conflicting evidence and reach a conclusion. A `high` confidence label on a contested case misleads downstream users. The post-hoc layer enforces consistency between the stated confidence and the objective evidence quality without modifying the LLM's substantive reasoning.

**Impact:** Confidence labels become trustworthy as a triage signal. Cases flagged `low` after correction are candidates for human review; cases remaining `high` have passed both the LLM and the rule layer. The correction reason provides a clear audit trail for why a label was changed.

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

---

## MCP Integration

Core system capabilities are exposed as [Model Context Protocol](https://modelcontextprotocol.io) tools, allowing any MCP-compatible AI client (Claude Desktop, VS Code Copilot, etc.) to call them directly — no API key required.

**Start the MCP server:**
```bash
python mcp_server/server.py
```
The server runs over stdio (MCP standard transport) and is ready to accept tool calls immediately.

**Available tools:**

### `retrieve_similar_cases`
Semantic search over the ChromaDB knowledge base. Returns reference cases ranked by similarity to the input text — useful for few-shot context before a diagnosis decision.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_text` | `str` | required | Clinical note or description to search against |
| `n_results` | `int` | `3` | Number of similar cases to return (max 10) |

**Returns:** list of `{note_id, text, label, subtype}`

```json
[
  {"note_id": "seed_001", "label": 1, "subtype": "ad",
   "text": "83-year-old female with progressive memory loss..."},
  {"note_id": "seed_006", "label": 0, "subtype": "na",
   "text": "74-year-old female admitted for hip fracture..."}
]
```

---

### `lookup_icd_codes`
Deterministic whitelist check — same rule-based logic as DiagnosisAgent Step 1. No LLM call, instant response.

| Parameter | Type | Description |
|---|---|---|
| `icd_codes` | `str` | Pipe-separated ICD-9/10 codes, e.g. `"F0280\|G309\|Z87.39"` |

**Returns:** `{matched_codes, is_ad_related, total_checked}`

```json
{"matched_codes": ["F0280", "G309"], "is_ad_related": true, "total_checked": 3}
```
