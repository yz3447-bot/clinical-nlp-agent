"""
rag/seed_data.py — Synthetic reference cases for the RAG knowledge base.

10 cases: 5 AD/ADRD positive (ad, vd, ftd subtypes), 5 negative.
Texts are short clinical summaries (50-100 words) representative of each class.
These are synthetic and do NOT contain real patient data.
"""

SEED_CASES = [
    # ── Positive: Alzheimer's disease (ad) ───────────────────────────────────
    {
        "note_id": "seed_001",
        "text": (
            "83-year-old female with a 4-year history of progressive memory loss. "
            "Daughter reports she repeats questions within minutes, cannot recall "
            "recent events, and became lost driving a familiar route. "
            "MMSE score 18/30. Neurological exam reveals mild word-finding difficulty. "
            "Currently prescribed donepezil 10 mg daily. Assessment: Alzheimer's disease, "
            "moderate stage. Plan: continue cholinesterase inhibitor, caregiver support."
        ),
        "label": 1,
        "subtype": "ad",
    },
    {
        "note_id": "seed_002",
        "text": (
            "77-year-old male admitted for urinary tract infection. Incidental finding: "
            "significant cognitive decline per family. Patient unable to state the year, "
            "does not recognize his daughter. Wife reports 3-year insidious onset of "
            "forgetfulness progressing to misidentifying family members. "
            "On memantine 10 mg BID at home. Discharge diagnosis includes Alzheimer's "
            "dementia as active problem requiring ongoing outpatient management."
        ),
        "label": 1,
        "subtype": "ad",
    },
    # ── Positive: vascular dementia (vd) ─────────────────────────────────────
    {
        "note_id": "seed_003",
        "text": (
            "71-year-old male with history of two ischemic strokes (2019, 2022) presenting "
            "with stepwise cognitive decline. Neuropsychological testing shows executive "
            "dysfunction and slowed processing speed disproportionate to memory loss. "
            "MRI demonstrates periventricular white matter hyperintensities and old lacunar "
            "infarcts. ICD-10 F0150 coded as active. Assessment: vascular dementia secondary "
            "to cerebrovascular disease. Rivastigmine initiated."
        ),
        "label": 1,
        "subtype": "vd",
    },
    # ── Positive: frontotemporal dementia (ftd) ───────────────────────────────
    {
        "note_id": "seed_004",
        "text": (
            "62-year-old female referred for behavioral changes over 2 years. Husband "
            "describes socially inappropriate comments, hyperphagia, and loss of empathy. "
            "No significant memory impairment. FDG-PET shows frontal hypometabolism. "
            "Neuropsychological testing reveals prominent frontal executive deficits with "
            "relatively preserved episodic memory. Diagnosis: behavioral variant "
            "frontotemporal dementia. Coded G3183. Started on SSRI for behavioral symptoms."
        ),
        "label": 1,
        "subtype": "ftd",
    },
    # ── Positive: non-specific dementia (nsd) ────────────────────────────────
    {
        "note_id": "seed_005",
        "text": (
            "88-year-old male nursing home resident admitted for pneumonia. Chart documents "
            "severe dementia of unspecified type — patient non-verbal, fully dependent for "
            "ADLs, does not recognize caregivers. ICD-10 F039 active on problem list. "
            "Family declines further workup given age and overall trajectory. Dementia "
            "management deferred to primary care; current admission focused on infection treatment."
        ),
        "label": 1,
        "subtype": "nsd",
    },
    # ── Negative cases ────────────────────────────────────────────────────────
    {
        "note_id": "seed_006",
        "text": (
            "74-year-old female admitted for hip fracture following mechanical fall. "
            "Alert and oriented x4 throughout admission. No cognitive complaints from "
            "patient or family. MMSE 29/30. Past medical history includes hypertension "
            "and type 2 diabetes. No history of dementia. No AD medications prescribed. "
            "Discharge to skilled nursing facility for rehabilitation. Dementia not "
            "mentioned in Assessment and Plan."
        ),
        "label": 0,
        "subtype": "na",
    },
    {
        "note_id": "seed_007",
        "text": (
            "68-year-old male presenting with acute delirium in the context of urosepsis. "
            "Mental status fluctuates; he is confused and intermittently agitated. "
            "Family confirms baseline is fully independent with intact cognition. "
            "Workup negative for primary neurological cause. Assessment: ICU delirium "
            "secondary to sepsis — NOT dementia. Resolved with treatment of underlying "
            "infection. No dementia diagnosis made."
        ),
        "label": 0,
        "subtype": "na",
    },
    {
        "note_id": "seed_008",
        "text": (
            "79-year-old female admitted for COPD exacerbation. Son mentions his mother "
            "sometimes forgets things but functional at home, manages finances independently. "
            "Neurology consulted and documents mild age-appropriate memory changes, "
            "not meeting criteria for MCI or dementia. No ICD codes for dementia applied. "
            "No AD medications. Discharge diagnosis: COPD exacerbation, resolved."
        ),
        "label": 0,
        "subtype": "na",
    },
    {
        "note_id": "seed_009",
        "text": (
            "81-year-old male with past medical history of dementia (remote diagnosis, "
            "not active per neurology re-evaluation 2023). Admitted for chest pain, "
            "found to have NSTEMI. Dementia listed in problem list as historical — "
            "no active management, no current AD medications, no cognitive symptoms "
            "documented this admission. Assessment and Plan focus entirely on cardiac "
            "management. Dementia coded as historical carry-forward only."
        ),
        "label": 0,
        "subtype": "na",
    },
    {
        "note_id": "seed_010",
        "text": (
            "66-year-old female presenting with depression and subjective memory complaints. "
            "Neuropsychological testing reveals attention and processing speed deficits "
            "consistent with major depressive disorder, not dementia. Family history notable "
            "for mother with Alzheimer's disease — patient anxious about her own risk. "
            "Assessment: pseudodementia secondary to depression. Started on sertraline. "
            "No dementia diagnosis. No AD medications. Follow-up in 3 months."
        ),
        "label": 0,
        "subtype": "na",
    },
]
