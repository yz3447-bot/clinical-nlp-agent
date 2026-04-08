"""
api.py — FastAPI wrapper for the AD/ADRD Horizontal Multi-Agent Pipeline.

Endpoints:
  POST /classify  — run the 3-agent + synthesizer pipeline on one clinical note
  GET  /health    — service liveness check
"""

import os
import time
import concurrent.futures
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from agents.diagnosis_agent import run_diagnosis_agent
from agents.medication_agent import run_medication_agent
from agents.clintext_agent import run_clintext_agent
from agents.synthesizer_agent import run_synthesizer_agent


# ─── Pydantic schemas ────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Full clinical note text")
    icd_codes: str = Field(
        default="",
        description="Pipe-separated ICD-9/10 codes (e.g. 'F0280|G309|3310'). "
                    "Leave empty if unavailable.",
    )


_LLM_COST_PER_CALL_USD = 0.002


class ClassifyResponse(BaseModel):
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
    discrepancy: Optional[str] = Field(
        None, description="Agent disagreements and how they were resolved, if any"
    )
    latency_ms: float = Field(..., description="Total pipeline wall-clock time in milliseconds")
    llm_calls: int = Field(..., description="Actual number of LLM calls made for this request")
    estimated_cost_usd: float = Field(..., description="Estimated API cost in USD")


class HealthResponse(BaseModel):
    status: str
    model: str
    agents: list[str]


# ─── App lifespan: initialise LLM once at startup ────────────────────────────

_llm: Optional[ChatGoogleGenerativeAI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm
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
    yield
    _llm = None


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
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["inference"])
def classify(request: ClassifyRequest):
    """
    Run the full 3-agent + synthesizer pipeline on a single clinical note.

    - Agents 1–3 run in parallel (ThreadPoolExecutor).
    - SynthesizerAgent runs serially after all three finish.
    """
    if _llm is None:
        raise HTTPException(status_code=503, detail="LLM not initialised — check GEMINI_API_KEY.")

    row = {
        "text": request.text,
        "all_icd_codes": request.icd_codes,
    }

    t0 = time.perf_counter()

    # ── Parallel: specialist agents ───────────────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_dx   = executor.submit(run_diagnosis_agent,  row, _llm)
        future_med  = executor.submit(run_medication_agent, row, _llm)
        future_clin = executor.submit(run_clintext_agent,   row, _llm)

        try:
            dx_result   = future_dx.result()
            med_result  = future_med.result()
            clin_result = future_clin.result()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    # ── Serial: synthesizer ───────────────────────────────────────────────────
    try:
        synth = run_synthesizer_agent(dx_result, med_result, clin_result, _llm, text=request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesizer error: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000

    # ── LLM call accounting ───────────────────────────────────────────────────
    # ClinTextAgent: always 1 call (LLM-only agent)
    calls_clin = 1
    # MedicationAgent: 0 if structured columns triggered, else 1
    calls_med = 0 if med_result.get("source") == "structured" else 1
    # DiagnosisAgent: 0 if no ICD codes matched whitelist, else 1
    calls_dx = 0 if not dx_result.get("matched_codes") else 1
    # SynthesizerAgent: 0 on consensus paths, 1 on LLM arbitration
    calls_synth = 0 if synth.get("synthesis_mode") in ("consensus_positive", "consensus_negative") else 1

    llm_calls = calls_clin + calls_med + calls_dx + calls_synth

    return ClassifyResponse(
        prediction=int(synth["final_prediction"]),
        subtype=synth["subtype"],
        confidence=synth["confidence"],
        synthesis_mode=synth["synthesis_mode"],
        causal_chain=synth["causal_chain"],
        contributing_agents=synth["contributing_agents"],
        discrepancy=synth.get("discrepancy") or None,
        latency_ms=round(latency_ms, 2),
        llm_calls=llm_calls,
        estimated_cost_usd=round(llm_calls * _LLM_COST_PER_CALL_USD, 4),
    )
