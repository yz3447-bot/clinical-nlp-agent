"""
mcp_server/server.py — MCP tool layer for the AD/ADRD CyberDoctor system.

Exposes two tools over the Model Context Protocol (stdio transport):
  - retrieve_similar_cases : semantic search over the ChromaDB case library
  - lookup_icd_codes        : whitelist check for AD/ADRD ICD-9/10 codes

Run standalone:
    python mcp_server/server.py

MCP clients (Claude Desktop, VS Code, etc.) connect via stdio.
No GEMINI_API_KEY required — these tools are purely local.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from agents.diagnosis_agent import AD_ICD_WHITELIST
from rag.knowledge_base import KnowledgeBase
from rag.seed_data import SEED_CASES


# ─── MCP server instance ──────────────────────────────────────────────────────

mcp = FastMCP("ad-cyberdoctor")


# ─── Knowledge base (lazy-initialised, seeds on first use) ───────────────────

_kb: KnowledgeBase | None = None


def _get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
        for case in SEED_CASES:
            _kb.add_case(
                note_id=case["note_id"],
                text=case["text"],
                label=case["label"],
                subtype=case["subtype"],
            )
    return _kb


# ─── Tool 1: retrieve_similar_cases ──────────────────────────────────────────

@mcp.tool()
def retrieve_similar_cases(query_text: str, n_results: int = 3) -> list[dict]:
    """Retrieve the most similar clinical cases from the AD/ADRD knowledge base.

    Uses ChromaDB vector similarity to find reference cases that resemble the
    provided clinical text. Useful for in-context learning before diagnosis.

    Args:
        query_text: Clinical note text or description to search against.
        n_results:  Number of similar cases to return (default 3, max 10).

    Returns:
        List of cases, each with:
          - note_id  (str)   : case identifier
          - text     (str)   : clinical summary
          - label    (int)   : 1 = AD/ADRD present, 0 = not present
          - subtype  (str)   : ad | vd | ftd | nsd | na
    """
    n_results = min(max(1, n_results), 10)
    kb = _get_kb()
    return kb.retrieve_similar(query_text, n_results=n_results)


# ─── Tool 2: lookup_icd_codes ─────────────────────────────────────────────────

@mcp.tool()
def lookup_icd_codes(icd_codes: str) -> dict:
    """Check which ICD codes in a pipe-separated list are on the AD/ADRD whitelist.

    Performs the same rule-based lookup used by DiagnosisAgent (Step 1) —
    no LLM call required, fully deterministic.

    Args:
        icd_codes: Pipe-separated ICD-9 or ICD-10 codes, e.g. "F0280|G309|Z87.39".
                   Dots and casing are normalised automatically.

    Returns:
        - matched_codes  (list[str]) : codes that matched the AD/ADRD whitelist
        - is_ad_related  (bool)      : True if at least one code matched
        - total_checked  (int)       : total number of codes in the input
    """
    raw_codes = [c.strip() for c in icd_codes.split("|") if c.strip()]

    def _normalise(code: str) -> str:
        return code.replace(".", "").strip().upper()

    matched = []
    for code in raw_codes:
        norm = _normalise(code)
        for wl_code in AD_ICD_WHITELIST:
            if norm == wl_code or norm.startswith(wl_code):
                matched.append(code)   # return original (un-normalised) form
                break

    return {
        "matched_codes": matched,
        "is_ad_related": len(matched) > 0,
        "total_checked": len(raw_codes),
    }


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
