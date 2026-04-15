"""
review_queue.py — SQLite-backed review queue for low-confidence predictions.

Table: review_queue
  id                   INTEGER PRIMARY KEY AUTOINCREMENT
  note_id              TEXT
  text_snippet         TEXT        (first 200 chars of clinical note)
  prediction           INTEGER     (0 or 1)
  confidence           TEXT        (high / medium / low)
  confidence_score_clin REAL
  confidence_score_med  REAL
  confidence_score_dx   REAL
  causal_chain         TEXT
  discrepancy          TEXT
  status               TEXT        (pending / approved / rejected)
  reviewer_comment     TEXT
  created_at           TEXT        (ISO-8601 UTC)
  reviewed_at          TEXT        (ISO-8601 UTC, NULL until reviewed)
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent / "outputs" / "review_queue.db"


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the review_queue table if it doesn't exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id               TEXT,
                text_snippet          TEXT,
                prediction            INTEGER,
                confidence            TEXT,
                confidence_score_clin REAL DEFAULT 0.0,
                confidence_score_med  REAL DEFAULT 0.0,
                confidence_score_dx   REAL DEFAULT 0.0,
                causal_chain          TEXT,
                discrepancy           TEXT,
                status                TEXT DEFAULT 'pending',
                reviewer_comment      TEXT,
                created_at            TEXT,
                reviewed_at           TEXT
            )
        """)
        conn.commit()


def add_to_queue(
    note_id: str,
    text: str,
    prediction: int,
    confidence: str,
    confidence_score_clin: float = 0.0,
    confidence_score_med: float = 0.0,
    confidence_score_dx: float = 0.0,
    causal_chain: str = "",
    discrepancy: str = "",
) -> int:
    """
    Insert a new case into the review queue.
    Returns the new row id.
    """
    init_db()
    snippet = text[:200] if text else ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO review_queue
                (note_id, text_snippet, prediction, confidence,
                 confidence_score_clin, confidence_score_med, confidence_score_dx,
                 causal_chain, discrepancy, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                note_id, snippet, prediction, confidence,
                confidence_score_clin, confidence_score_med, confidence_score_dx,
                causal_chain, discrepancy, now,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_pending_cases() -> list[dict]:
    """Return all rows with status = 'pending', newest first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM review_queue WHERE status = 'pending' ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_reviewed_cases() -> list[dict]:
    """Return all rows with status != 'pending', newest reviewed first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM review_queue WHERE status != 'pending' ORDER BY reviewed_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def approve_case(row_id: int, comment: str = "") -> None:
    """Mark a case as approved."""
    _update_status(row_id, "approved", comment)


def reject_case(row_id: int, comment: str = "") -> None:
    """Mark a case as rejected."""
    _update_status(row_id, "rejected", comment)


def get_all_cases() -> list[dict]:
    """Return all rows ordered by created_at DESC."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM review_queue ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def bulk_import_from_csv(csv_path) -> int:
    """
    Import all records from predictions_c.csv into the review queue.
    Skips note_ids that already exist in the queue.
    Returns the number of newly inserted rows.
    """
    import pandas as pd
    path = Path(csv_path)
    if not path.exists():
        return 0

    init_db()

    df = pd.read_csv(path, dtype=str).fillna("")

    # Gather existing note_ids to avoid duplicates
    with _connect() as conn:
        existing = {
            r[0]
            for r in conn.execute("SELECT note_id FROM review_queue").fetchall()
        }

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    inserted = 0

    for _, row in df.iterrows():
        note_id = str(row.get("note_id", ""))
        if not note_id or note_id in existing:
            continue

        try:
            prediction = int(float(row.get("final_prediction", 0)))
        except (ValueError, TypeError):
            prediction = 0

        confidence = str(row.get("confidence", "low"))
        causal_chain = str(row.get("causal_chain", "") or row.get("reason", ""))
        discrepancy = str(row.get("discrepancy", ""))

        # confidence_score_* columns may be absent in old CSVs
        def _safe_float(val, default=0.0):
            try:
                return float(val) if val not in ("", None) else default
            except (ValueError, TypeError):
                return default

        cs_clin = _safe_float(row.get("confidence_score_clin", 0.0))
        cs_med  = _safe_float(row.get("confidence_score_med", 0.0))
        cs_dx   = _safe_float(row.get("confidence_score_dx", 0.0))

        text_snippet = str(row.get("text", ""))[:200]

        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO review_queue
                    (note_id, text_snippet, prediction, confidence,
                     confidence_score_clin, confidence_score_med, confidence_score_dx,
                     causal_chain, discrepancy, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    note_id, text_snippet, prediction, confidence,
                    cs_clin, cs_med, cs_dx,
                    causal_chain, discrepancy, now,
                ),
            )
            conn.commit()

        existing.add(note_id)
        inserted += 1

    return inserted


def _update_status(row_id: int, status: str, comment: str) -> None:
    init_db()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _connect() as conn:
        conn.execute(
            """
            UPDATE review_queue
            SET status = ?, reviewer_comment = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (status, comment, now, row_id),
        )
        conn.commit()
