# v1_baseline — First Trusted Eval Run

## Run Info

| Field | Value |
|-------|-------|
| run_id | `1a4bc982-caa2-4a17-af15-5dfc19286ed1` |
| eval date | 2026-05-01 |
| predictions file | predictions.csv (87 rows) |
| dataset | data_train.csv — 87 labeled rows evaluated (label ∈ {0,1}) |

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8966 (78/87) |
| Sensitivity (Recall) | 0.8868 |
| PPV (Precision) | 0.9400 |
| Specificity | 0.9118 |
| F1 | 0.9126 |

### Confusion Matrix

|  | Pred 1 | Pred 0 |
|--|--------|--------|
| **GT 1** | TP = 47 | FN = 6 |
| **GT 0** | FP = 3 | TN = 31 |

### Error Buckets

| Bucket | Count |
|--------|-------|
| Total errors | 9 |
| False Positives | 3 |
| False Negatives | 6 |
| Contradictory errors (had discrepancy) | 3 |
| Low-confidence errors | 5 |
| Consensus path errors | 1 |
| All cases with discrepancy | 21 |

CI gate: Sensitivity 0.8868 < 0.90 threshold → **FAIL**

## System State at Time of This Run

### RAG / Knowledge Base
- 10 synthetic seed cases only (rag/seed_data.py)
- ChromaDB seeded once at startup; no real EHR cases

### MedicationAgent
- Phase 1: binary flag columns (denepezil, memantine, tacrine, rivastigmine, galantamine)
- Binary columns were dead code — all zeros in this dataset
- Effective path: 100% LLM free-text scan (Phase 2)
- Section extraction (`_extract_med_sections`): present

### DiagnosisAgent
- Section extraction (`_extract_dx_sections`): present (including Brief Hospital Course pattern)
- Confidence scoring: `is_current=False → 0.1` (non-zero, incorrect)
- `high → 0.8`, `medium → 0.5`, `low → 0.3`

### ClinTextAgent
- Confidence scoring: 2-tier quantity (≥3 → 0.4, else → 0.2)
- Source quality: assessment/plan/discharge → +0.3, hpi/pmh → +0.1
- Negation penalty: any negation quote → −0.3 (fires too aggressively)

### SynthesizerAgent
- `consensus_positive` triggered on **med AND clin** only (dx not required)
- `_apply_confidence_correction`: early-exit rules (Rule 1/2/3), did not stack
- `consensus_negative`: no confidence correction applied

### Pipeline
- `med_flags` dict passed through (5 binary columns, all zeros in practice)
- No `ad_med_annotation` parameter

## What Changed in v2 (next run)

- `populate_kb.py`: KB expanded to 134 cases (10 synthetic + 124 real EHR from data_train)
- `rag/knowledge_base.py`: `clear()` method added
- `agents/medication_agent.py`: human annotation column replaces binary flags; new confidence tiers
- `agents/clintext_agent.py`: 4-tier evidence quantity; hpi +0.2, pmh +0.05; negation only if >50% quotes
- `agents/diagnosis_agent.py`: `is_current=False → 0.0`; `high → 0.9`, `medium → 0.6`
- `agents/synthesizer_agent.py`: consensus_positive requires all 3 agents; stacking Rules A/B/C
- `pipeline.py`: `ad_med_annotation` replaces `med_flags`
- `eval/eval_harness.py`: subtype normalization (`fd→ftd`, `other→nsd`)
