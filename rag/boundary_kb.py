"""
rag/boundary_kb.py — ChromaDB-backed retrieval of boundary judgment principles.

Unlike the old KnowledgeBase (similar clinical cases), BoundaryKnowledgeBase
stores expert-curated boundary rules derived from error analysis.
Retrieval is triggered only when ClinTextAgent detects a boundary signal;
the retrieved principles are injected as a second-pass prompt.
"""

from pathlib import Path

import chromadb

from logger import get_logger

logger = get_logger(__name__)

_DB_PATH        = Path(__file__).parent / "chroma_db"
_COLLECTION     = "boundary_principles"
_DIST_THRESHOLD = 0.7   # L2 distance; results above this are too dissimilar to use


def _principle_to_text(p: dict) -> str:
    return (
        f"boundary_type: {p['boundary_type']}\n"
        f"clinical_features: {p['clinical_features']}\n"
        f"principle: {p['principle']}\n"
        f"key_evidence: {p['key_evidence']}\n"
        f"correct_conclusion: {p['correct_conclusion']}\n"
        f"common_mistake: {p['common_mistake']}"
    )


class BoundaryKnowledgeBase:
    def __init__(self, collection_name: str = _COLLECTION):
        self._client = chromadb.PersistentClient(path=str(_DB_PATH))
        self._col    = self._client.get_or_create_collection(collection_name)

    def populate(self, principles: list[dict]) -> int:
        """Embed principles into the collection. Idempotent — skips existing entries.

        Returns the number of newly added principles.
        """
        added = 0
        for p in principles:
            pid = p["boundary_type"]
            if self._col.get(ids=[pid])["ids"]:
                continue  # already exists
            self._col.add(
                ids=[pid],
                documents=[_principle_to_text(p)],
                metadatas=[{k: v for k, v in p.items() if k != "boundary_type"}],
            )
            added += 1
        if added:
            logger.info("[BoundaryKB] Added %d new principles (total: %d)", added, self._col.count())
        return added

    def retrieve(self, query: str, n_results: int = 2) -> list[dict]:
        """Return up to n_results principles most relevant to query.

        Filters out results with L2 distance > _DIST_THRESHOLD.
        Returns list of full principle dicts (boundary_type + all metadata fields).
        """
        count = self._col.count()
        if count == 0:
            return []
        n = min(n_results, count)
        results = self._col.query(query_texts=[query], n_results=n)

        principles = []
        for i, pid in enumerate(results["ids"][0]):
            dist = results["distances"][0][i]
            if dist > _DIST_THRESHOLD:
                continue
            meta = results["metadatas"][0][i]
            principles.append({"boundary_type": pid, **meta})
        return principles

    def count(self) -> int:
        return self._col.count()
