"""
schemas.py — Shared Pydantic data models for all inter-agent communication.

Every agent must return a validated instance of its output schema.
This ensures structural mismatches are caught at module boundaries,
not silently propagated through the pipeline.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class DiagnosisAgentOutput(BaseModel):
    """Output schema for DiagnosisAgent (ICD code specialist, weight 1)."""
    dx_found: bool
    matched_codes: list[str] = Field(default_factory=list)
    is_current_diagnosis: bool = False
    confidence: Literal["high", "medium", "low"] = "low"
    confidence_score: float = 0.0
    reasoning: str = ""
    assessment: str = ""
    weight: Literal[1] = 1


class MedicationAgentOutput(BaseModel):
    """Output schema for MedicationAgent (pharmacist, weight 3)."""
    meds_found: bool
    medications: list[str] = Field(default_factory=list)
    status: Literal["current", "historical", "refused", "mentioned", "none"] = "none"
    source: Literal["structured", "annotation", "text", "none"] = "none"
    confidence: Literal["high", "medium", "low"] = "low"
    confidence_score: float = 0.0
    reasoning: str = ""
    assessment: str = ""
    weight: Literal[3] = 3


class ClinTextAgentOutput(BaseModel):
    """Output schema for ClinTextAgent (neurologist, weight 3)."""
    symptoms_found: bool
    evidence_list: list[str] = Field(default_factory=list)
    reasoning: str = ""
    assessment: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    confidence_score: float = 0.0
    weight: Literal[3] = 3


class SynthesizerOutput(BaseModel):
    """Output schema for SynthesizerAgent (attending physician, final judgment)."""
    final_prediction: Literal[0, 1]
    subtype: Literal["ad", "vd", "ftd", "nsd", "na"]
    confidence: Literal["high", "medium", "low"]
    synthesis_mode: Literal["consensus_positive", "consensus_negative", "llm_arbitration"]
    total_score: int = 0
    contributing_agents: list[str] = Field(default_factory=list)
    causal_chain: str = ""
    discrepancy: str | None = None
    overruled_agents: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence_score_clin: float = 0.0
    confidence_score_med: float = 0.0
    confidence_score_dx: float = 0.0
    confidence_correction: str | None = None


class PipelineResult(BaseModel):
    """
    Final output schema for the full pipeline.
    Used by both api.py and main_c.py (batch).
    This is the single source of truth for what the system outputs.
    """
    note_id: str = ""
    subject_id: str = ""
    ground_truth: str = ""
    ground_truth_subtype: str = ""

    # Per-agent signal flags
    dx_found: bool = False
    meds_found: bool = False
    symptoms_found: bool = False

    # Final decision
    final_prediction: Literal[0, 1]
    subtype: Literal["ad", "vd", "ftd", "nsd", "na"]
    confidence: Literal["high", "medium", "low"]
    synthesis_mode: Literal["consensus_positive", "consensus_negative", "llm_arbitration"]

    # Evidence and reasoning
    contributing_agents: list[str] = Field(default_factory=list)
    causal_chain: str = ""
    discrepancy: str | None = None
    overruled_agents: list[str] = Field(default_factory=list)
    reason: str = ""

    # Per-agent confidence scores
    confidence_score_clin: float = 0.0
    confidence_score_med: float = 0.0
    confidence_score_dx: float = 0.0
    confidence_correction: str | None = None

    # Observability fields
    latency_ms: float = 0.0
    llm_calls: int = 0
    estimated_cost_usd_proxy: float = 0.0
    degraded_mode: bool = False
    agent_errors: list[str] = Field(default_factory=list)

    # Run-level traceability
    run_id: str = ""
    timestamp: str = ""
    text_snippet: str = ""
