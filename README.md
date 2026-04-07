# AD/ADRD CyberDoctor — Horizontal Multi-Agent System

Automated identification of Alzheimer's Disease and Alzheimer's Disease Related Dementias (AD/ADRD) from EHR discharge summaries using a horizontal multi-agent LLM pipeline.

---

## Architecture Overview

```
EHR Record (full text + ICD codes)
        │
        ├──────────────────────────────────────────────┐
        │                        │                     │
        ▼                        ▼                     ▼
┌───────────────┐   ┌────────────────────┐   ┌──────────────────┐
│ ClinTextAgent │   │  MedicationAgent   │   │ DiagnosisAgent   │
│  (Layer 1)    │   │    (Layer 2)       │   │   (Layer 3)      │
│  weight: +3   │   │    weight: +3      │   │   weight: +1     │
│               │   │                   │   │                  │
│  LLM reads    │   │  Phase 1: struct   │   │  Step 1: rule    │
│  full note as │   │  Phase 2: LLM full │   │  ICD matching    │
│  neurologist  │   │  text as pharmacist│   │  Step 2: LLM ctx │
└───────┬───────┘   └────────┬───────────┘   └────────┬─────────┘
        │                    │                         │
        └────────────────────┴─────────────────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │    SynthesizerAgent     │
                        │   (Attending Physician) │
                        │                        │
                        │  Always LLM-based.      │
                        │  Receives full note +   │
                        │  all 3 specialist       │
                        │  reports. Can overrule  │
                        │  any agent.             │
                        └────────────┬───────────┘
                                     │
                                     ▼
                           final_prediction (0/1)
                           subtype (ad/vd/ftd/nsd/na)
                           confidence, causal_chain
                           overruled_agents, reason
```

All three agents run **in parallel** via `ThreadPoolExecutor`. The Synthesizer runs serially after all three complete.

---

## Effective Features

Based on data analysis, only three feature types carry meaningful signal in this dataset:

| Feature | Source | Notes |
|---|---|---|
| Full clinical text | `text` column | Primary evidence — average 11,200 chars, max 45,700 |
| ICD codes | `all_icd_codes` column | 99.5% coverage for positives, but 12.3% false trigger on negatives |
| Subtype annotation | `adrd_dx_subtype` related columns | Complete labels: ad / vd / nsd / na |

Features that are **not used**:
- Structured medication columns — all zero in this dataset
- Radiology notes — unrelated to AD (orthopedic/chest imaging)
- Demographics — no independent diagnostic signal

---

## Design Philosophy

### 1. Three Independent Specialist Perspectives

Each agent reads the same complete clinical note but through a different professional lens:

- **ClinTextAgent** — neurologist: what cognitive symptoms are present right now?
- **MedicationAgent** — pharmacist: is the patient currently prescribed AD medications?
- **DiagnosisAgent** — coding specialist: are the ICD codes reflecting active or historical disease?

No agent votes. No majority rule. The Synthesizer weighs all three perspectives with full clinical reasoning and can overrule any of them.

### 2. Clinical Judgment, Not Keyword Extraction

The central mistake to avoid: treating "AD mentioned in the note" as equivalent to "this patient has AD."

A clinical note may mention AD in non-diagnostic contexts:

- **Family history** — "mother had Alzheimer's" → irrelevant to patient's diagnosis
- **Differential being ruled out** — "dementia vs delirium — more likely delirium" → negative signal
- **Historical background** — "PMH: dementia (not active this admission)" → not current
- **Refused medications** — "patient declined donepezil" → no active management

Every agent prompt explicitly instructs the LLM to exclude these cases and explain what was excluded in the `reasoning` field.

### 3. Full Text, No Truncation

Early versions truncated text to 3,000–6,000 characters. This was harmful for this dataset:

| Metric | Value |
|---|---|
| Average note length | ~11,200 chars |
| Maximum note length | ~45,700 chars |
| Key evidence location | Often in the second half (discharge diagnosis, Assessment & Plan) |

All agents receive the **complete** clinical text. Gemini 2.5 Flash supports a 1M token context window — truncation provides no benefit and actively harms recall.

### 4. Current vs Historical Distinction

A recurring source of false positives is historical diagnosis coded into the record without being active during the current admission. Each agent is designed to make this distinction:

- **ClinTextAgent** — excludes "historical conditions not active this admission"; requires evidence from the current encounter
- **MedicationAgent** — classifies medication status as `current / historical / refused / mentioned / none`; only `current` sets `meds_found=true`
- **DiagnosisAgent** — after ICD match, LLM reads the full note to set `is_current_diagnosis`; codes appearing only in PMH without active management are flagged as historical

### 5. Synthesizer Has Full Clinical Authority

The Synthesizer receives the complete clinical note plus all three formatted specialist reports (including each agent's reasoning and assessment). It:

- Evaluates each agent's findings independently
- Identifies and resolves contradictions between agents
- Explains which agent to trust more and why, when they disagree
- Can explicitly overrule any agent, with the overruled agent listed in `overruled_agents`
- Outputs both the AD/ADRD diagnosis and the dementia subtype

This design handles the common case where agents contradict each other — e.g., ICD codes present (DiagnosisAgent positive) but the note describes an orthopedic admission with dementia only in the PMH (ClinTextAgent and MedicationAgent negative). The Synthesizer resolves this with clinical reasoning rather than mechanical voting.

### 6. Subtype as Supplementary Output

Subtype classification (`ad / vd / ftd / nsd / na`) is produced by the Synthesizer alongside the primary diagnosis. It is written to the output CSV as a reference field but is **not included in performance evaluation** — the primary metric is binary AD/ADRD identification accuracy.

---

## Agent Details

### ClinTextAgent (`agents/clintext_agent.py`)

**Persona:** Experienced clinical neurologist specializing in dementia

**Input:** Full EHR text  
**Output:** `symptoms_found`, `confidence`, `evidence_list` (exact quotes from note), `reasoning`, `assessment`

**Prompt focus:** Chief Complaint, HPI, physical/neurological exam, Assessment & Plan, Discharge Diagnosis, Discharge Summary. Evidence must be verbatim quotes. The `reasoning` field explains what was included and what was excluded and why.

**Confidence:**
- `high` — physician explicitly documents active AD/ADRD, or 2+ distinct symptom categories with quotes
- `medium` — implicit cognitive symptoms without explicit diagnosis
- `low` — single vague or ambiguous mention

---

### MedicationAgent (`agents/medication_agent.py`)

**Persona:** Clinical pharmacist specializing in dementia care

**Medications covered:** Donepezil (Aricept), Memantine (Namenda), Rivastigmine (Exelon), Galantamine (Razadyne), Tacrine, and any cholinesterase inhibitor for cognitive symptoms

**Phase 1 (rule-based):** Checks structured binary columns. Returns immediately if any flag is 1. In this dataset these are all zero, so Phase 1 never triggers.

**Phase 2 (LLM):** Primary path for this dataset. LLM reads the full note and classifies medication status:

| Status | Meaning | Sets `meds_found` |
|---|---|---|
| `current` | Actively prescribed during this admission or at discharge | Yes |
| `historical` | Taken in the past, now discontinued | No |
| `refused` | Offered but patient declined | No |
| `mentioned` | Discussed but not prescribed | No |
| `none` | No AD medication found | No |

**Output:** `meds_found`, `medications`, `status`, `source`, `confidence`, `reasoning`, `assessment`

---

### DiagnosisAgent (`agents/diagnosis_agent.py`)

**Persona:** Medical coding specialist and diagnostician

**Step 1 (rule-based, no LLM):** Matches `all_icd_codes` against a whitelist of ~50 ICD-9 and ICD-10 codes using prefix matching. If no codes match, returns immediately (no LLM call).

**ICD-9 whitelist:** 2900, 29010–29043, 2940, 29410–29421, 3310–33182, 3349, 34830, 2930, 2931  
**ICD-10 whitelist:** F0280, F0281, F0390, F0391, G30xx, G31xx, F02xx, F03xx, F0150, F0151

**Step 2 (LLM, only if codes matched):** LLM reads the full note to determine whether the matched codes reflect a current active diagnosis or a historical one carried forward:

| Finding | `is_current_diagnosis` | `confidence` |
|---|---|---|
| ICD codes + active dementia management in note | `true` | `high` |
| ICD codes + no mention in clinical note | `true` | `medium` |
| ICD codes + "history of" without current management | `false` | `low` |

**Output:** `dx_found`, `matched_codes`, `is_current_diagnosis`, `confidence`, `reasoning`, `assessment`

---

### SynthesizerAgent (`agents/synthesizer_agent.py`)

**Persona:** Senior attending physician and dementia specialist

**Input:** Full clinical note + formatted reports from all three specialist agents (including each agent's reasoning and assessment)

**Mode:** Always LLM-based. Rule-based fallback only if the LLM call fails.

**Task (two parts):**

*Part 1 — AD/ADRD Diagnosis:*  
Evidence hierarchy used by the LLM: explicit physician documentation > current AD medications > current ICD codes with note support > documented cognitive symptoms > ICD codes without note support > single vague mention

*Part 2 — Subtype Classification:*

| Subtype | Clinical profile |
|---|---|
| `ad` | Progressive memory loss, language problems, gradual onset |
| `vd` | Stepwise decline, stroke/CVD history, focal neurological signs |
| `ftd` | Behavioral changes, language problems, younger onset, frontal symptoms |
| `nsd` | Dementia present but insufficient evidence for specific subtype |
| `na` | No AD/ADRD (final_prediction = 0) |

Subtype is always `na` when `final_prediction = 0`.

**Output:** `final_prediction`, `subtype`, `confidence`, `contributing_agents`, `causal_chain`, `discrepancy`, `overruled_agents`, `reason`

---

## Output Schema

Results are written to `outputs/predictions_c.csv`:

| Column | Description |
|---|---|
| `note_id` | Patient note identifier |
| `subject_id` | Patient identifier |
| `ground_truth` | Manual label (1=AD/ADRD, 0=No ADRD) |
| `ground_truth_subtype` | Manual subtype label (ad/vd/nsd/na) |
| `dx_found` | DiagnosisAgent: ICD code matched (0/1) |
| `meds_found` | MedicationAgent: current AD med found (0/1) |
| `symptoms_found` | ClinTextAgent: current AD symptoms found (0/1) |
| `final_prediction` | Synthesizer diagnosis (0/1) |
| `subtype` | Synthesizer subtype (ad/vd/ftd/nsd/na) — output only, not evaluated |
| `confidence` | Synthesizer confidence (high/medium/low) |
| `contributing_agents` | JSON list of agents that found positive evidence |
| `causal_chain` | Synthesizer step-by-step reasoning |
| `discrepancy` | Agent disagreements and how resolved |
| `overruled_agents` | JSON list of agents overruled by Synthesizer |
| `reason` | One-sentence final summary |

Records with `ground_truth = -1` (uncertain) are skipped. Processing supports **checkpoint/resume**: previously processed `note_id`s are skipped on restart.

---

## Performance Metrics

`compute_metrics()` evaluates binary AD/ADRD identification only:

- **Accuracy** — overall correct predictions
- **Sensitivity** — true positive rate (recall for AD/ADRD cases)
- **PPV** — positive predictive value (precision)

Subtype classification is not evaluated — `subtype` is a supplementary output field.

---

## File Structure

```
ad_cyberdoctor_horizontal/
├── main_c.py                    # Entry point: data loading, parallel execution, metrics
├── test_run.py                  # Verbose test on first 5 records with validation checks
├── visualize.py                 # Generates outputs/report.html (interactive report)
├── agents/
│   ├── clintext_agent.py        # Layer 1: neurologist clinical judgment
│   ├── medication_agent.py      # Layer 2: pharmacist medication check (LLM primary)
│   ├── diagnosis_agent.py       # Layer 3: ICD rule match + LLM current/historical check
│   ├── synthesizer_agent.py     # Final: attending physician synthesis + subtype
│   └── cogtest_agent.py         # Retired (not imported or called)
├── data/
│   └── data_test.csv            # Input EHR data
└── outputs/
    ├── predictions_c.csv        # Agent predictions
    └── report.html              # Interactive visualization report
```

---

## Running the Pipeline

```bash
export GEMINI_API_KEY=your_key_here

# Full run
python main_c.py

# Verbose test on first 5 records
python test_run.py

# Generate interactive HTML report
python visualize.py
```

---

## Key Dataset Properties

| Property | Value |
|---|---|
| Total records | 100 |
| Labeled (0 or 1) | 87 (after removing uncertain -1) |
| Positive cases (AD/ADRD) | ~54% |
| Average note length | ~11,200 characters |
| Maximum note length | ~45,700 characters |
| ICD coverage (positive cases) | ~99.5% |
| ICD false trigger rate (negative cases) | ~12.3% |
| Structured medication columns | All zero — LLM text scan is primary path |
| Subtype distribution | nsd: 40, na: 35, ad: 8, vd: 4 |

---

## LLM Configuration

| Parameter | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| Provider | Google AI (via LangChain `ChatGoogleGenerativeAI`) |
| Temperature | 0 (deterministic) |
| Context window | 1,000,000 tokens |
| LLM calls per record | 4 (ClinText + Medication + Diagnosis + Synthesizer) |
| Rate limit buffer | 30-second pause between records |
