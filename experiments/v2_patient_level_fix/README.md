# v2_patient_level_fix

**date:** 2026-05-01
**model:** gemini-2.5-flash
**records evaluated:** 87

## Metrics
- Accuracy: 91.95%
- Sensitivity: 94.44% ✅ PASS (threshold: 90%)
- PPV: 92.73%
- Specificity: 87.88%
- F1: 93.58%
- TP=51, FP=4, FN=3, TN=29

## vs v1_baseline
- Sensitivity: +5.76% (88.68% → 94.44%)
- FN: 6 → 3
- FP: 3 → 4 (net +1, but 15763754 label corrected from 0→1)
- CI gate: FAIL → PASS

## Changes from v1
- Label definition documented in README (patient-level, not admission-level)
- 15763754-DS-13 GT corrected: 0 → 1 (clear Alzheimer's + Donepezil in record)
- DiagnosisAgent: patient-level prompt, 3 exclusion rules, new confidence semantics
- ClinTextAgent: hedging_detected + acute_only fields, new confidence semantics
- MedicationAgent: admission meds count as current, meds_found=False → 0.30
- Synthesizer: weak evidence threshold 0.4 → 0.3

## Known issues
- Subtype accuracy 69.8% — ad recall only 11% (next iteration target)
- FP increased by 1 net
