"""
rag/boundary_cases.py — Expert-curated boundary judgment principles for AD/ADRD detection.

Each principle was derived from error analysis of known FN/FP patterns.
Loaded into BoundaryKnowledgeBase at startup and retrieved when ClinTextAgent
detects signals of a boundary case (hedging language, acute-only symptoms, etc.).
"""

BOUNDARY_PRINCIPLES = [
    {
        "boundary_type": "acute_delirium_with_chronic_dementia",
        "clinical_features": "acute AMS or delirium on admission, PMH mentions dementia, discharge mental status clear or improved, no AD medications",
        "principle": "Delirium superimposed on dementia is a recognized clinical entity. Acute delirium resolving at discharge does NOT negate underlying chronic dementia. A single mention of dementia in PMH is sufficient evidence for a positive classification.",
        "key_evidence": "Any PMH mention of dementia or Alzheimer's, cognitive symptoms on admission even if resolved",
        "correct_conclusion": "positive",
        "common_mistake": "Classifying as negative because delirium resolved and discharge mental status is clear",
    },
    {
        "boundary_type": "discharge_normal_chronic_dementia",
        "clinical_features": "discharge mental status clear and coherent, PMH has dementia, current admission for unrelated reason",
        "principle": "Normal mental status at discharge does not negate a pre-existing chronic dementia diagnosis. Patients return to their baseline. The question is patient-level AD/ADRD status, not admission-level active management.",
        "key_evidence": "PMH mention of dementia, prior ICD codes for AD/ADRD",
        "correct_conclusion": "positive",
        "common_mistake": "Using clear discharge mental status as evidence against chronic dementia",
    },
    {
        "boundary_type": "hedging_language_only",
        "clinical_features": "possible dementia, suspected cognitive impairment, rule out dementia, recommend neurocognitive evaluation",
        "principle": "Uncertain language is NOT confirmatory evidence. Possible or suspected dementia means the physician has not yet confirmed the diagnosis. Classify as uncertain or negative, not positive.",
        "key_evidence": "Need at least one definitive physician statement or confirmed ICD code with text support",
        "correct_conclusion": "negative or uncertain",
        "common_mistake": "Treating possible/suspected dementia as equivalent to confirmed dementia",
    },
    {
        "boundary_type": "no_ad_medications",
        "clinical_features": "no donepezil, memantine, rivastigmine, or galantamine found anywhere in the note",
        "principle": "Absence of AD medications is NEUTRAL evidence, not negative evidence. Many confirmed AD/ADRD patients are not prescribed these medications due to advanced disease, contraindications, family preference, or economic reasons.",
        "key_evidence": "Do not use medication absence to override strong clinical or ICD evidence",
        "correct_conclusion": "neutral — do not use as negative evidence",
        "common_mistake": "Downgrading positive classification because no AD medications were found",
    },
    {
        "boundary_type": "acute_delirium_icd_only",
        "clinical_features": "only F05 acute delirium or G92 toxic encephalopathy ICD codes, no separate chronic dementia ICD, no PMH mention of dementia",
        "principle": "Acute delirium codes without any documentation of underlying chronic dementia represent an exclusion case. These are reversible acute conditions, not chronic AD/ADRD.",
        "key_evidence": "Must have separate documentation of chronic dementia — either ICD code or PMH mention",
        "correct_conclusion": "negative",
        "common_mistake": "Classifying as positive because acute AMS was documented",
    },
    {
        "boundary_type": "boundary_icd_g312",
        "clinical_features": "only G312 alcohol-related neurodegeneration ICD code, note describes acute alcohol events, Wernicke encephalopathy, or alcohol withdrawal",
        "principle": "G312 alone without explicit chronic dementia documentation is an exclusion case. Acute alcohol-related neurological events are not equivalent to chronic AD/ADRD. Must have explicit mention of chronic alcohol-related dementia.",
        "key_evidence": "Need explicit chronic dementia description beyond acute alcohol events",
        "correct_conclusion": "negative unless explicit chronic dementia documented",
        "common_mistake": "Treating G312 as equivalent to AD/ADRD without chronic dementia documentation",
    },
    {
        "boundary_type": "death_carryover_icd",
        "clinical_features": "patient died or withdrew care this admission, discharge diagnosis contains only acute cause of death, no cognitive content in note body",
        "principle": "ICD codes carried forward to a death record without any cognitive documentation represent historical codes, not active chronic conditions. The note must have some cognitive content to support a positive classification.",
        "key_evidence": "Some cognitive documentation in note body beyond just ICD codes",
        "correct_conclusion": "negative",
        "common_mistake": "Classifying as positive solely because AD/ADRD ICD code exists on a death record",
    },
]
