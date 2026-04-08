"""
rag/knowledge_base.py — ChromaDB-backed case retrieval for ClinTextAgent.

Stores clinical case summaries as vector embeddings (ChromaDB default model).
Persisted locally at rag/chroma_db/ — not committed to git.
"""

from pathlib import Path

import chromadb

_DB_PATH = Path(__file__).parent / "chroma_db"
_COLLECTION_NAME = "ad_cases"


class KnowledgeBase:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=str(_DB_PATH))
        self._col = self._client.get_or_create_collection(_COLLECTION_NAME)

    def add_case(self, note_id: str, text: str, label: int, subtype: str) -> None:
        """Add a single clinical case to the knowledge base.

        Silently skips if note_id already exists (safe to call on restart).
        """
        existing = self._col.get(ids=[note_id])
        if existing["ids"]:
            return
        self._col.add(
            ids=[note_id],
            documents=[text],
            metadatas=[{"label": label, "subtype": subtype}],
        )

    def retrieve_similar(self, text: str, n_results: int = 3) -> list[dict]:
        """Return the n most similar cases to the given text.

        Returns:
            List of dicts: {note_id, text, label, subtype}
        """
        count = self._col.count()
        if count == 0:
            return []
        # Never request more results than items in the collection
        n = min(n_results, count)
        results = self._col.query(query_texts=[text], n_results=n)
        cases = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            cases.append({
                "note_id": results["ids"][0][i],
                "text":    doc,
                "label":   int(meta["label"]),
                "subtype": str(meta["subtype"]),
            })
        return cases

    def get_stats(self) -> int:
        """Return the total number of cases stored in the knowledge base."""
        return self._col.count()
