# AD/ADRD CyberDoctor — Horizontal Multi-Agent System

**Detects Alzheimer's Disease and related dementias (AD/ADRD) from hospital EHR discharge summaries using three parallel AI specialist agents with rule-based confidence scoring and post-hoc correction, achieving 88.68% sensitivity on 87 labeled cases (v1_baseline) while eliminating the false positives caused by naive keyword matching or ICD code lookups alone.**

> **Research Prototype Notice**
> This system is a research prototype developed for academic purposes.
> It is not validated for clinical use and must not be used to inform
> real medical decisions. All predictions should be reviewed by qualified
> clinicians before any clinical action is taken.

## Results

| Dataset | Records | Labeled | AD/ADRD Positive | Avg Note Length |
|---|---|---|---|---|
| MIMIC-style EHR discharge summaries | 100 | 87 (excl. uncertain) | ~54% | ~11,200 chars |

Metrics are printed automatically after each run. Reproduce with:
```bash
python main_c.py
```

## Label Definition

本系统的判断目标是：
**这份出院病历是否提供了足够可信的证据，
证明这个病人有慢性 AD/ADRD。**

不是判断"本次住院是否在 active management AD/ADRD"。
历史诊断、PMH 记录、入院用药里的 AD 药物，全部算有效证据。

### 正例（GT=1）标准

满足以下任一条，且标注员判断证据指向真实慢性认知退化：

1. AD/ADRD 相关 ICD 编码存在，且病历文本中有任何支撑性描述
   （包括 PMH 一句 "dementia"、HPI 提及认知障碍、
   查体发现定向力障碍、家属描述记忆力下降等）

2. 病历文本中医生明确记录了 AD/ADRD 或慢性认知障碍诊断

3. 当前用药或入院用药包含 AD 专用药物
   （donepezil / memantine / rivastigmine / galantamine）

### 负例（GT=0）标准

以下三种情况即便有 AD/ADRD ICD 编码也判为负例：

- **排除一（急性谵妄）**：认知相关 ICD 仅为急性谵妄（F05）
  或急性脑病（G92），且文本中无慢性认知退化的任何描述

- **排除二（死亡携带码）**：患者本次因急症死亡或撤治，
  出院诊断不含 AD/ADRD，文本中完全无认知相关内容

- **排除三（边界 ICD）**：匹配码仅为 G312
  （酒精性神经退化），且文本只描述急性酒精相关事件，
  无慢性痴呆的明确描述

### Uncertain（GT=-1）标准

病历中有认知相关描述但措辞为 possible / suspected /
rule out，或证据极其稀少、无法确信。

### 重要说明

- 本次住院主诊非 AD → 不影响判断
- 出院时精神状态正常 → 不否定慢性诊断
- 无 AD 药物 → 中性，不是阴性证据
- 急性谵妄解决 → 不排除底层慢性痴呆

## Architecture

```
EHR Record (full text + ICD codes)
         │
         ├─────────────────────┬────────────────────┐
         ▼                     ▼                    ▼
 ┌───────────────┐   ┌──────────────────┐  ┌────────────────┐
 │ ClinTextAgent │   │ MedicationAgent  │  │ DiagnosisAgent │
 │  neurologist  │   │   pharmacist     │  │  coding spec.  │
 │  LLM +        │   │  hybrid (struct  │  │  rule-based +  │
 │  boundary RAG │   │    + LLM)        │  │  LLM context   │
 │               │   │    + LLM)        │  │  LLM context   │
 └───────┬───────┘   └────────┬─────────┘  └───────┬────────┘
         │   confidence_score  │  confidence_score   │  confidence_score
         └────────────────────┴─────────────────────┘
                               │  (parallel via ThreadPoolExecutor)
                               ▼
                    ┌──────────────────────┐
                    │   SynthesizerAgent   │
                    │  attending physician │
                    │                      │
                    │  ① consensus_positive│  all 3 agents pos
                    │     → skip LLM       │  → pred=1, high conf
                    │  ② consensus_negative│  all three negative
                    │     → skip LLM       │  → pred=0, high conf
                    │  ③ llm_arbitration   │  mixed signals
                    │     → LLM + scores   │  → LLM resolves
                    │  ④ post-hoc rules    │  3 forced downgrades
                    └──────────┬───────────┘
                               ▼
         final_prediction · subtype · confidence · synthesis_mode
         causal_chain · contributing_agents · discrepancy · overruled_agents
         confidence_score_clin/med/dx · confidence_correction · agent_findings
```

| Agent | Role | Method | Confidence Score Logic |
|---|---|---|---|
| **ClinTextAgent** | Chronic AD/ADRD evidence in full note | LLM as neurologist; boundary-aware RAG triggers a second LLM call when hedging/acute signals detected, injecting 1–2 expert judgment principles derived from error analysis | Confidence = certainty in own conclusion; strong ≥2 → 0.90; hedging-only → 0.35; acute-only → 0.50 |
| **MedicationAgent** | Active AD prescriptions | Phase 1: structured columns; Phase 2: LLM full-text scan | annotation/structured=0.9 · current text (recognized drug)=0.75 · current text (unrecognized)=0.6 · historical/refused=0.2 · mentioned=0.05 · not found=0.0 |
| **DiagnosisAgent** | ICD codes — current vs historical | Step 1: rule-based whitelist match (early exit if no match); Step 2: LLM context check | not found=0.0 · historical=0.0 · current+high=0.9 · current+medium=0.6 · current+low=0.3 |
| **SynthesizerAgent** | Final diagnosis + subtype | Consensus early exit or LLM arbitration with agent scores; deterministic fallback | Post-hoc correction: 3 forced downgrade rules |

## Quickstart

**Prerequisites:** Python 3.10+, a Google Gemini API key, and your EHR data CSV in `data/`.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export GEMINI_API_KEY=your_key_here   # Windows: set GEMINI_API_KEY=your_key_here

# 3. Run the full batch pipeline
python main_c.py

# 4. (Optional) Test on 5 records with verbose output
python test_run.py

# 5. (Optional) Generate standalone interactive HTML report
python visualize.py
# → outputs/report.html
```

**Streamlit Frontend:**

```bash
# Terminal 1 — FastAPI backend (requires GEMINI_API_KEY)
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit UI
streamlit run app.py
# → http://localhost:8501
```

Three pages:
- **Classify** — select a dataset record or paste a note; renders agent cards with ✓/✗, confidence progress bars, expandable evidence details, and a colour-coded verdict block
- **Review Queue** — bulk-import from `predictions_c.csv`, or auto-populated after every classify call (all classification results are added unconditionally, not filtered by confidence or discrepancy); approve / reject with comments
- **Eval Dashboard** — metric cards, confusion matrix, error-bucket bar chart, per-agent confidence histograms, contradictory-case table

**REST API (FastAPI):**

Live demo: https://clinical-nlp-agent.onrender.com/docs

```bash
# Self-host
uvicorn api:app --host 0.0.0.0 --port 8000

# Single-record inference
curl -X POST https://clinical-nlp-agent.onrender.com/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with progressive memory loss...", "icd_codes": "F0280|G309"}'
```

**Full API response schema:**
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "prediction": 1,
  "subtype": "ad",
  "confidence": "high",
  "synthesis_mode": "llm_arbitration",
  "causal_chain": "ClinTextAgent found explicit dementia documentation in A&P...",
  "contributing_agents": ["ClinTextAgent", "DiagnosisAgent"],
  "overruled_agents": [],
  "discrepancy": null,
  "reason": "Alzheimer's dementia confirmed by clinical documentation and ICD coding.",
  "latency_ms": 4821.3,
  "llm_calls": 3,
  "estimated_cost_usd_proxy": 0.006,
  "confidence_score_clin": 0.7,
  "confidence_score_med": 0.0,
  "confidence_score_dx": 0.8,
  "confidence_correction": null,
  "agent_findings": {
    "symptoms_found": true,
    "evidence_list": ["progressive memory loss documented in A&P..."],
    "clin_reasoning": "Clear dementia documented in assessment and plan.",
    "meds_found": false,
    "medications": [],
    "med_status": "none",
    "med_source": "none",
    "dx_found": true,
    "matched_codes": ["F0280"],
    "is_current_diagnosis": true,
    "dx_reasoning": "F0280 appears as active diagnosis in the problem list."
  }
}
```

## Evaluation

The eval harness reads `outputs/predictions_c.csv` — **no pipeline re-run required**.

```bash
# Basic eval — no API key needed
python eval/eval_harness.py

# Full eval with LLM as judge (requires GEMINI_API_KEY)
python eval/eval_harness.py --run-llm-judge
```

**Results on test set (87 labeled cases) — v1_baseline, `run_id 1a4bc982-caa2-4a17-af15-5dfc19286ed1`:**

| Metric | Value |
|---|---|
| Accuracy | **89.66%** (78 / 87) |
| Sensitivity | **88.68%** |
| PPV (Precision) | **94.00%** |
| Specificity | **91.18%** |
| F1 | **91.26%** |
| Cases with discrepancy | **21 / 87 (24%)** — agents disagreed; resolved by Synthesizer |

**Error buckets:**

| Bucket | Count | Definition |
|---|---|---|
| False Positives | 3 | Predicted AD/ADRD, actually negative |
| False Negatives | 6 | Predicted no AD/ADRD, actually positive |
| Contradictory errors | 3 | Error cases with agent discrepancy — core focus for LLM judge |
| Low-confidence errors | 5 | Error cases where `confidence == "low"` after post-hoc correction |
| Consensus errors | 1 | All three agents agreed but prediction was wrong |

**LLM as judge** scores each contradictory case on three dimensions (0–1):
- `reasoning_clarity` — decision traceable to specific evidence?
- `contradiction_handling` — overrule justification sufficient?
- `evidence_consistency` — conclusion consistent with cited evidence?

**CI gate:** Runs on every push. If `predictions_c.csv` exists and sensitivity drops below **0.90**, the build fails.

## Engineering Design Decisions

### 1. Parallel Agent Execution
**What:** All three specialist agents run simultaneously via `ThreadPoolExecutor(max_workers=3)`.

**Why:** Each agent makes an independent LLM call (~2–5 s). They share no state, so running sequentially would triple latency with zero benefit.

**Impact:** ~3× faster per record. Total latency is bounded by the slowest agent, not their sum.

### 2. Evidence Fusion via Rule-Based Confidence Scores
**What:** Each agent computes a deterministic `confidence_score` (0–1) from objective evidence characteristics, independently of what the LLM self-reports. These scores are passed as explicit prompt signals to the Synthesizer, which references them when weighing evidence but retains full judgment authority — there are no fixed numeric weights or hard votes.

**Why:** A current AD medication prescription means a physician has *already diagnosed* the patient and started treatment — the strongest possible indirect confirmation. ICD codes alone are weakest: they are routinely carried forward from previous admissions without reflecting the current encounter. Rule-based scores make these distinctions explicit and inspectable without locking the Synthesizer into a rigid arithmetic formula.

**Impact:** The system mirrors real clinical reasoning. Medication evidence produces a high confidence score that pushes the Synthesizer toward a positive prediction; ICD codes alone produce a low score that the Synthesizer can and does override when the note context warrants it.

### 3. Consensus-Based Early Exit
**What:** Before invoking the LLM Synthesizer, the system checks for agreement: if all three agents (MedicationAgent, ClinTextAgent, and DiagnosisAgent) are positive → `consensus_positive` (predict 1, skip LLM); if all three agents are negative → `consensus_negative` (predict 0, skip LLM). Only mixed signals trigger `llm_arbitration`.

**Why:** When strong independent signals agree, LLM arbitration adds cost and latency without changing the outcome. Consensus cases — both majority-positive and all-negative — are the clearest, so the deterministic early exit is correct and cheaper.

**Impact:** Reduces Synthesizer LLM calls on unambiguous records. Frees the LLM budget for genuinely contested cases. `synthesis_mode` in every response makes it auditable which path was taken.

### 4. Contradiction Detection and Auditability
**What:** The Synthesizer explicitly detects disagreements between agents, records which agents were overruled, and writes the full reasoning chain to `causal_chain` and `overruled_agents`.

**Why:** Medical AI must be auditable. A black-box "prediction: 1" is not actionable for a clinician. Every decision needs to explain *why* — especially when agents contradict each other (e.g., ICD codes present but note shows an orthopedic admission with dementia only in past history).

**Impact:** Full decision traceability. Reviewers can inspect exactly which agent was trusted, which was overruled, and why — for every single prediction.

### 5. Graceful Fallback — Zero Silent Failures
**What:** If the Synthesizer's LLM call fails or returns unparseable output, the system falls back to a deterministic rule-based decision (medication → ICD + symptoms → symptoms → ICD → negative, in priority order).

**Why:** Silent failures are the worst outcome in a medical system. The fallback trades LLM reasoning for a safe, explainable rule that always produces a valid output.

**Impact:** The pipeline never crashes mid-run or returns a null prediction. Fallback results are labeled `[Fallback — LLM failed]` in `causal_chain` so reviewers know which records to re-examine.

### 6. Cost-Optimized ICD Lookup
**What:** DiagnosisAgent runs a rule-based ICD whitelist match *before* any LLM call. If no AD/ADRD codes are present, it returns immediately with zero API cost.

**Why:** ~40% of records have no relevant ICD codes. Sending those to an LLM to confirm "nothing found" wastes quota and adds latency.

**Impact:** ~40% reduction in LLM calls for DiagnosisAgent. The LLM is only invoked when there is actual coded evidence to evaluate (current vs. historical decision).

### 7. Single LLM Initialization via Lifespan Pattern
**What:** In the FastAPI service, the Gemini LLM client and ChromaDB knowledge base are initialized once at startup via FastAPI's `lifespan` context manager — not recreated per request.

**Why:** LLM client creation involves authentication, SDK setup, and connection overhead (~100–300 ms). ChromaDB requires loading its on-disk index. Rebuilding per request wastes these costs under concurrent load.

**Impact:** Both resources are shared across all requests. Per-request latency stays predictable and resource contention is avoided.

### 8. Rule-Based Confidence Scoring
**What:** Each agent computes a `confidence_score` (0–1) deterministically from objective evidence characteristics, independently of the LLM's self-reported confidence:

| Agent | Score Logic |
|---|---|
| ClinTextAgent | ≥5 evidence items → +0.5; ≥2 → +0.35; else → +0.15; +0.3 Assessment/Plan/Discharge; +0.2 HPI; +0.05 PMH; −0.2 if >50% quotes negated |
| MedicationAgent | 0.9 annotation/structured · 0.75 current text (recognized drug) · 0.6 current text (unrecognized) · 0.2 historical/refused · 0.05 mentioned · 0.0 not found |
| DiagnosisAgent | 0.0 not found · 0.0 historical (non-current) · 0.9/0.6/0.3 current + high/medium/low LLM confidence |

**Why:** LLM self-reported confidence is unreliable — models frequently express high confidence regardless of actual evidence quality. Rule-based scores are reproducible, inspectable, and cannot be inflated by confident-sounding but unsupported LLM text.

**Impact:** The Synthesizer receives objective evidence quality signals alongside each agent's narrative. The LLM judge and downstream systems can use these scores independently of the LLM's own assessment.

### 9. Post-hoc Confidence Correction
**What:** After the Synthesizer produces its output, a deterministic rule layer applies forced confidence downgrades before the result is returned. Three rules (applied in order, early-return on first match):

1. Overruled agents present **and** current confidence is `high` → cap at `medium`
2. All contributing agents have `confidence_score < 0.4` → force to `low`
3. Positive prediction (`final_prediction=1`) with no contributing agents → force to `low`

All three rules are evaluated in order and stack — all applicable rules apply before returning. Each downgrade appends its reason to `confidence_correction`.

**Why:** LLMs tend toward overconfidence when synthesizing conflicting evidence. A `high` label on a contested case misleads downstream users and reviewers.

**Impact:** Confidence labels become trustworthy as a triage signal. Cases remaining `high` have passed both the LLM and the rule layer. The correction reason provides a clear audit trail.

### 10. RAG-Augmented Clinical Context
**What:** ClinTextAgent retrieves the 3 most similar clinical cases from a ChromaDB vector knowledge base and prepends them as few-shot context to its prompt. RAG failure never blocks the agent. Both the API server (via FastAPI lifespan) and the batch pipeline (`main_c.py`) now initialize the KnowledgeBase at startup and seed it with the same 10 reference cases, so both modes receive identical RAG context. The current seed data consists of 10 synthetically authored cases (5 AD/ADRD positive across subtypes, 5 negative). Production deployments should replace or extend these with real, annotated clinical cases to maximize retrieval quality.

**Why:** Gemini-2.5-Flash with no context can misclassify ambiguous cases — dementia mentioned only in past history, or delirium mimicking cognitive decline. Concrete reference cases anchor the model's reasoning to clinically validated examples.

**Impact:** The LLM receives labeled examples at inference time without fine-tuning. The knowledge base is extensible: add a confirmed case with `KnowledgeBase.add_case()` and it is immediately available for future retrievals across both the API and batch pipeline.

## File Structure

```
ad_cyberdoctor_horizontal/
├── main_c.py                   # Batch pipeline: load CSV → agents → predictions_c.csv → metrics
├── api.py                      # FastAPI service: POST /classify, GET /health + JSONL logging
├── pipeline.py                 # Shared pipeline core: parallel dispatch, timeouts, PipelineResult
├── schemas.py                  # Pydantic data models for all inter-agent communication
├── logger.py                   # Centralised logging configuration (LOG_LEVEL env var)
├── app.py                      # Streamlit frontend: Classify / Review Queue / Eval Dashboard
├── review_queue.py             # SQLite review queue (add, approve, reject, bulk import)
├── visualize.py                # Standalone interactive HTML report generator
├── requirements.txt
├── render.yaml                 # Render.com deployment config
├── agents/
│   ├── clintext_agent.py       # Neurologist: symptom evidence + RAG + confidence_score
│   ├── medication_agent.py     # Pharmacist: structured fast-path + LLM scan + confidence_score
│   ├── diagnosis_agent.py      # Coder: ICD whitelist + LLM currency check + confidence_score
│   └── synthesizer_agent.py    # Attending: consensus exit / LLM arbitration / fallback / post-hoc
├── rag/
│   ├── knowledge_base.py       # ChromaDB PersistentClient wrapper (add, retrieve, stats)
│   └── seed_data.py            # 10 synthetic seed cases (5 AD+ subtypes, 5 negative)
├── mcp_server/
│   └── server.py               # MCP tools: retrieve_similar_cases, lookup_icd_codes
├── eval/
│   ├── eval_harness.py         # Metrics, error buckets, LLM judge, CI sensitivity gate
│   ├── llm_judge.py            # LLM as judge scoring for contradictory cases
│   └── report_generator.py     # Self-contained HTML eval report
├── tests/
│   ├── test_synthesizer.py     # Unit tests for SynthesizerAgent logic (Pydantic fixtures)
│   └── test_pipeline_contract.py  # Schema contracts: PipelineResult construction, EVIDENCE_COLUMNS coverage
├── .github/workflows/ci.yml    # flake8 lint + import checks + sensitivity gate (≥0.90)
├── .streamlit/config.toml      # Dark medical theme
├── data/
│   └── data_test.csv           # Input EHR data
└── outputs/
    ├── predictions_c.csv       # Per-record predictions with full reasoning (checkpoint/resume)
    ├── agent_evidence.csv      # Per-agent raw evidence linked to predictions_c.csv by note_id + run_id
    ├── review_queue.db         # SQLite review queue
    └── report.html             # Interactive visualization report
```

## Output Schema

### `outputs/predictions_c.csv` (batch pipeline)

| Column | Description |
|---|---|
| `note_id` / `subject_id` | EHR record identifiers |
| `final_prediction` | **1** = AD/ADRD present, **0** = not present |
| `subtype` | `ad` / `vd` / `ftd` / `nsd` / `na` |
| `confidence` | `high` / `medium` / `low` |
| `synthesis_mode` | `consensus_positive` / `consensus_negative` / `llm_arbitration` |
| `causal_chain` | Synthesizer's step-by-step reasoning |
| `contributing_agents` | Agents with positive findings (JSON array) |
| `overruled_agents` | Agents the Synthesizer disagreed with (JSON array) |
| `discrepancy` | Agent conflicts and how resolved |
| `reason` | One-sentence final summary from the Synthesizer |
| `dx_found` / `meds_found` / `symptoms_found` | Per-agent binary signal flags |
| `confidence_score_clin` / `confidence_score_med` / `confidence_score_dx` | Rule-based evidence quality scores (0–1) |
| `confidence_correction` | Reason if post-hoc rules downgraded Synthesizer confidence |
| `latency_ms` | Total wall-clock time for the record |
| `llm_calls` | Actual LLM calls made (1–4 depending on path taken) |
| `estimated_cost_usd_proxy` | Estimated API cost at $0.002 per call |
| `run_id` | UUID shared across all records in one invocation |
| `timestamp` | UTC timestamp of prediction |
| `text_snippet` | First 200 characters of the clinical note |

Processing supports **checkpoint/resume**: records already in `predictions_c.csv` are skipped on restart.

### `outputs/agent_evidence.csv` (batch pipeline)

Stores the raw per-agent outputs for every processed record, enabling post-hoc evidence audits without re-running the pipeline.

| Column | Description |
|---|---|
| `note_id` | Join key to `predictions_c.csv` |
| `run_id` | Join key to `predictions_c.csv` (identifies the batch run) |
| `clin_evidence_list` | JSON array of clinical text quotes extracted by ClinTextAgent |
| `clin_reasoning` | ClinTextAgent's full reasoning narrative |
| `clin_assessment` | ClinTextAgent's one-sentence assessment |
| `med_medications` | JSON array of AD medications detected by MedicationAgent |
| `med_status` | Medication status: `current` / `historical` / `refused` / `mentioned` / `none` |
| `med_source` | Where the medication was found: `structured` / `text` / `none` |
| `med_reasoning` | MedicationAgent's full reasoning narrative |
| `med_assessment` | MedicationAgent's one-sentence assessment |
| `dx_matched_codes` | JSON array of ICD codes matched to the AD/ADRD whitelist |
| `dx_is_current_diagnosis` | Whether matched codes reflect a current (not historical) diagnosis |
| `dx_reasoning` | DiagnosisAgent's full reasoning narrative |
| `dx_assessment` | DiagnosisAgent's one-sentence assessment |

Link records between the two files using `note_id` and `run_id` together.

### Review Queue (`outputs/review_queue.db`)

Each case stores: `note_id`, `text_snippet` (200 chars), `prediction`, `confidence`, `confidence_score_clin/med/dx`, `causal_chain`, `discrepancy`, `status` (pending / approved / rejected), `reviewer_comment`, `created_at`, `reviewed_at`.

## LLM Configuration

| Parameter | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| Temperature | `0` (deterministic) |
| Context window | 1,000,000 tokens (full notes, no truncation) |
| LLM calls per record | 1–4 (ClinTextAgent always calls LLM; consensus = 1–3; arbitration = 4; DiagnosisAgent skips if no ICD match) |
| Rate limit buffer | 30 s between records (batch mode) |

## MCP Integration

Core capabilities are exposed as [Model Context Protocol](https://modelcontextprotocol.io) tools, callable from any MCP-compatible client (Claude Desktop, VS Code Copilot, etc.) — no API key required.

```bash
python mcp_server/server.py   # runs over stdio (MCP standard transport)
```

### `retrieve_similar_cases`
Semantic search over the ChromaDB knowledge base.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_text` | `str` | required | Clinical note or description to search against |
| `n_results` | `int` | `3` | Number of cases to return (max 10) |

**Returns:** list of `{note_id, text, label, subtype}`

```json
[
  {"note_id": "seed_001", "label": 1, "subtype": "ad",
   "text": "83-year-old female with progressive memory loss..."},
  {"note_id": "seed_006", "label": 0, "subtype": "na",
   "text": "74-year-old female admitted for hip fracture..."}
]
```

### `lookup_icd_codes`
Deterministic whitelist check — same rule-based logic as DiagnosisAgent Step 1.

| Parameter | Type | Description |
|---|---|---|
| `icd_codes` | `str` | Pipe-separated ICD-9/10 codes, e.g. `"F0280\|G309\|Z87.39"` |

**Returns:** `{matched_codes, is_ad_related, total_checked}`

```json
{"matched_codes": ["F0280", "G309"], "is_ad_related": true, "total_checked": 3}
```
