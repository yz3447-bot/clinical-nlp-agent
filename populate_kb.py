"""
populate_kb.py — Seed ChromaDB with real cases from data_train.csv.

Selection logic:
  Positive (target ~90):
    - "ad"  (all 41)   — primary Alzheimer's class, take all
    - "vd"  (all  7)   — vascular dementia, small class, take all
    - "ftd" (all  2)   — includes "fd" normalised → "ftd"
    - "nsd" (top 40)   — prioritise med-annotated first, then longest text
    - Excluded: label=1 but subtype="other" (ambiguous) or "na" (contradictory)

  Negative (target ~34):
    - All 9 with AD ICD codes   — hardest false-positive scenario for RAG
    - Top 25 remaining          — sorted by text length (more informative notes)

Text stored in ChromaDB: first 800 chars of each note.
  Rationale: seed cases are 50-100 word summaries; full 11k notes would bloat
  every ClinTextAgent prompt by ~33k chars (3 retrieved cases × 11k).
  800 chars covers demographics + HPI onset, sufficient for similarity search.

Run:
    python populate_kb.py
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from rag.knowledge_base import KnowledgeBase
from rag.seed_data import SEED_CASES

DATA_PATH = Path(__file__).parent / "data" / "data_train.csv"

_SUBTYPE_NORMALIZE = {"fd": "ftd", "other": "nsd"}

_ANNOTATION_MED_KEYWORDS = frozenset({
    "donepezil", "aricept", "memantine", "namenda",
    "rivastigmine", "exelon", "galantamine", "razadyne", "tacrine",
})

_AD_ICD_PREFIXES = (
    "F02", "F03", "G30", "G31",
    "290", "291", "2940", "3310", "3311", "3312", "3316", "3349",
)

_TEXT_EXCERPT_LEN = 800  # chars stored per case in ChromaDB


def _has_ad_icd(icd_str: str) -> bool:
    codes = [c.strip().replace(".", "").upper() for c in icd_str.split("|") if c.strip()]
    return any(code.startswith(pfx) for code in codes for pfx in _AD_ICD_PREFIXES)


def _has_med_annotation(val: str) -> bool:
    v = val.lower()
    return any(kw in v for kw in _ANNOTATION_MED_KEYWORDS)


def select_cases(df: pd.DataFrame) -> list[dict]:
    label_col   = next(c for c in df.columns if "is AD/ADRD" in c)
    subtype_col = next(c for c in df.columns if "is subtype" in c)
    med_col     = next(c for c in df.columns if "AD/ADRD medications" in c)

    df = df.copy()
    df["_label"]       = df[label_col].str.strip()
    df["_subtype"]     = (
        df[subtype_col].str.strip().str.lower()
                       .map(lambda v: _SUBTYPE_NORMALIZE.get(v, v))
    )
    df["_has_med"]     = df[med_col].apply(_has_med_annotation)
    df["_text_len"]    = df["text"].str.len()
    df["_has_ad_icd"]  = df["all_icd_codes"].apply(_has_ad_icd)

    records: list[dict] = []
    seen_ids: set[str]  = set()

    def _add(row, subtype_override=None):
        nid = str(row["note_id"])
        if nid in seen_ids:
            return
        seen_ids.add(nid)
        records.append({
            "note_id": f"train_{nid}",
            "text":    str(row["text"])[:_TEXT_EXCERPT_LEN].strip(),
            "label":   1 if row["_label"] == "1" else 0,
            "subtype": subtype_override or row["_subtype"],
        })

    # ── Positive cases ────────────────────────────────────────────────────────
    # Exclude contradictory rows (label=1 but subtype="other" or ambiguous "na")
    pos_valid = df[
        (df["_label"] == "1") &
        (~df["_subtype"].isin(["other", "na"]))
    ]

    for subtype in ("ad", "vd", "ftd"):
        group = pos_valid[pos_valid["_subtype"] == subtype]
        for _, row in group.iterrows():
            _add(row)

    # nsd: top 40 by (has_med DESC, text_len DESC)
    nsd = (
        pos_valid[pos_valid["_subtype"] == "nsd"]
        .sort_values(["_has_med", "_text_len"], ascending=[False, False])
        .head(40)
    )
    for _, row in nsd.iterrows():
        _add(row)

    # ── Negative cases ────────────────────────────────────────────────────────
    neg = df[df["_label"] == "0"]

    # All negatives with AD ICD codes
    for _, row in neg[neg["_has_ad_icd"]].iterrows():
        _add(row)

    # Top-25 remaining negatives by text length
    neg_plain = (
        neg[~neg["_has_ad_icd"]]
        .sort_values("_text_len", ascending=False)
        .head(25)
    )
    for _, row in neg_plain.iterrows():
        _add(row)

    return records


def main() -> None:
    print(f"Loading {DATA_PATH.name} ...")
    df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
    cases = select_cases(df)

    pos_cases = [c for c in cases if c["label"] == 1]
    neg_cases = [c for c in cases if c["label"] == 0]
    subtype_dist = dict(Counter(c["subtype"] for c in pos_cases))

    print(f"\nSelected {len(cases)} real cases:")
    print(f"  Positive : {len(pos_cases)}  — subtypes: {subtype_dist}")
    print(f"  Negative : {len(neg_cases)}")

    print("\nResetting ChromaDB collection ...")
    kb = KnowledgeBase()
    kb.clear()

    print(f"Adding {len(SEED_CASES)} synthetic seed cases ...")
    for case in SEED_CASES:
        kb.add_case(
            note_id=case["note_id"],
            text=case["text"],
            label=case["label"],
            subtype=case["subtype"],
        )

    print(f"Adding {len(cases)} real cases (text excerpts: {_TEXT_EXCERPT_LEN} chars each) ...")
    for case in cases:
        kb.add_case(
            note_id=case["note_id"],
            text=case["text"],
            label=case["label"],
            subtype=case["subtype"],
        )

    total = kb.get_stats()
    print(f"\nKB final size : {total} cases")
    print(f"  Synthetic   : {len(SEED_CASES)}")
    print(f"  Real (train): {len(cases)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
