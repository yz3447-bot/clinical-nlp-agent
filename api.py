"""
api.py — FastAPI wrapper for the AD/ADRD Horizontal Multi-Agent Pipeline.

Endpoints:
  POST /classify  — run the 3-agent + synthesizer pipeline on one clinical note
  GET  /health    — service liveness check
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from pipeline import run_pipeline
from rag.knowledge_base import KnowledgeBase
from rag.seed_data import SEED_CASES


# ─── Logging ─────────────────────────────────────────────────────────────────

_LOG_PATH = Path(__file__).parent / "logs" / "requests.jsonl"


def _ensure_log_dir():
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _append_log(record: dict):
    _ensure_log_dir()
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _count_requests() -> int:
    if not _LOG_PATH.exists():
        return 0
    with open(_LOG_PATH, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ─── Pydantic schemas ────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Full clinical note text")
    icd_codes: str = Field(
        default="",
        description="Pipe-separated ICD-9/10 codes (e.g. 'F0280|G309|3310'). "
                    "Leave empty if unavailable.",
    )


class AgentFindings(BaseModel):
    """Raw per-agent findings returned alongside the synthesizer result."""
    # ClinTextAgent
    symptoms_found: bool
    evidence_list: list[str]
    clin_reasoning: str
    # MedicationAgent
    meds_found: bool
    medications: list[str]
    med_status: str
    med_source: str
    # DiagnosisAgent
    dx_found: bool
    matched_codes: list[str]
    is_current_diagnosis: bool
    dx_reasoning: str


class ClassifyResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for this request")
    prediction: int = Field(..., description="1 = AD/ADRD present, 0 = not present")
    subtype: str = Field(..., description="ad | vd | ftd | nsd | na")
    confidence: str = Field(..., description="high | medium | low")
    synthesis_mode: str = Field(
        ..., description="consensus_positive | consensus_negative | llm_arbitration"
    )
    causal_chain: str = Field(..., description="Step-by-step reasoning from the synthesizer")
    contributing_agents: list[str] = Field(
        ..., description="Agents whose findings supported the final decision"
    )
    overruled_agents: list[str] = Field(
        default_factory=list,
        description="Agents overruled by the Synthesizer, with reasons",
    )
    discrepancy: Optional[str] = Field(
        None, description="Agent disagreements and how they were resolved, if any"
    )
    reason: str = Field("", description="One-sentence final summary from the Synthesizer")
    latency_ms: float = Field(..., description="Total pipeline wall-clock time in milliseconds")
    llm_calls: int = Field(..., description="Actual number of LLM calls made for this request")
    estimated_cost_usd_proxy: float = Field(..., description="Estimated API cost in USD")
    confidence_score_clin: float = Field(..., description="Rule-based confidence score from ClinTextAgent (0–1)")
    confidence_score_med: float = Field(..., description="Rule-based confidence score from MedicationAgent (0–1)")
    confidence_score_dx: float = Field(..., description="Rule-based confidence score from DiagnosisAgent (0–1)")
    confidence_correction: Optional[str] = Field(
        None, description="Reason if post-processing rules downgraded Synthesizer confidence"
    )
    agent_findings: Optional[AgentFindings] = Field(
        None, description="Raw per-agent findings (evidence, medications, ICD codes)"
    )


class HealthResponse(BaseModel):
    status: str
    model: str
    agents: list[str]
    total_requests: int
    knowledge_base_size: int


# ─── App lifespan: initialise LLM once at startup ────────────────────────────

_llm: Optional[ChatGoogleGenerativeAI] = None
_kb:  Optional[KnowledgeBase] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _kb
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set. "
            "Export it before starting: export GEMINI_API_KEY=your_key"
        )
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )
    # Initialise ChromaDB and load seed cases (skips duplicates on restart)
    _kb = KnowledgeBase()
    for case in SEED_CASES:
        _kb.add_case(
            note_id=case["note_id"],
            text=case["text"],
            label=case["label"],
            subtype=case["subtype"],
        )
    yield
    _llm = None
    _kb  = None


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AD/ADRD CyberDoctor API",
    description="Horizontal multi-agent system for Alzheimer's / ADRD detection from EHR notes.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness check — confirms the service and LLM are ready."""
    return HealthResponse(
        status="ok" if _llm is not None else "degraded",
        model="gemini-2.5-flash",
        agents=["ClinTextAgent", "MedicationAgent", "DiagnosisAgent", "SynthesizerAgent"],
        total_requests=_count_requests(),
        knowledge_base_size=_kb.get_stats() if _kb is not None else 0,
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["inference"])
def classify(request: ClassifyRequest):
    """
    Run the full 3-agent + synthesizer pipeline on a single clinical note.
    All execution logic is delegated to pipeline.run_pipeline().
    """
    if _llm is None:
        raise HTTPException(status_code=503, detail="LLM not initialised — check GEMINI_API_KEY.")

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        pipeline_result, agent_detail, _ = run_pipeline(
            text=request.text,
            icd_codes=request.icd_codes,
            llm=_llm,
            kb=_kb,
            include_detail=True,
        )
    except Exception as exc:
        _append_log({
            "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "request_id":   request_id,
            "input_length": len(request.text),
            "error":        str(exc),
            "latency_ms":   round((time.perf_counter() - t0) * 1000, 2),
        })
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    findings = AgentFindings(
        symptoms_found       = pipeline_result.symptoms_found,
        evidence_list        = agent_detail["clin"]["evidence_list"],
        clin_reasoning       = agent_detail["clin"]["reasoning"],
        meds_found           = pipeline_result.meds_found,
        medications          = agent_detail["med"]["medications"],
        med_status           = agent_detail["med"]["status"],
        med_source           = agent_detail["med"]["source"],
        dx_found             = pipeline_result.dx_found,
        matched_codes        = agent_detail["dx"]["matched_codes"],
        is_current_diagnosis = agent_detail["dx"]["is_current_diagnosis"],
        dx_reasoning         = agent_detail["dx"]["reasoning"],
    )

    response = ClassifyResponse(
        request_id=request_id,
        prediction=int(pipeline_result.final_prediction),
        subtype=pipeline_result.subtype,
        confidence=pipeline_result.confidence,
        synthesis_mode=pipeline_result.synthesis_mode,
        causal_chain=pipeline_result.causal_chain,
        contributing_agents=list(pipeline_result.contributing_agents),
        overruled_agents=list(pipeline_result.overruled_agents),
        discrepancy=pipeline_result.discrepancy,
        reason=pipeline_result.reason,
        latency_ms=pipeline_result.latency_ms,
        llm_calls=pipeline_result.llm_calls,
        estimated_cost_usd_proxy=pipeline_result.estimated_cost_usd_proxy,
        confidence_score_clin=pipeline_result.confidence_score_clin,
        confidence_score_med=pipeline_result.confidence_score_med,
        confidence_score_dx=pipeline_result.confidence_score_dx,
        confidence_correction=pipeline_result.confidence_correction,
        agent_findings=findings,
    )

    _append_log({
        "timestamp":                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "request_id":               request_id,
        "input_length":             len(request.text),
        "icd_codes":                request.icd_codes,
        "prediction":               response.prediction,
        "subtype":                  response.subtype,
        "confidence":               response.confidence,
        "synthesis_mode":           response.synthesis_mode,
        "llm_calls":                response.llm_calls,
        "estimated_cost_usd_proxy": response.estimated_cost_usd_proxy,
        "latency_ms":               response.latency_ms,
        "has_discrepancy":          response.discrepancy is not None,
        "contributing_agents":      list(response.contributing_agents),
        "overruled_agents":         list(response.overruled_agents),
        "confidence_score_clin":    response.confidence_score_clin,
        "confidence_score_med":     response.confidence_score_med,
        "confidence_score_dx":      response.confidence_score_dx,
        "confidence_correction":    response.confidence_correction,
        "error":                    None,
    })

    return response
