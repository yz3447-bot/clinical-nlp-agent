"""
tests/test_pipeline_contract.py — Schema and contract tests for the pipeline layer.

Tests cover:
  - PipelineResult can be constructed with only required Literal fields
  - Every column in EVIDENCE_COLUMNS is assigned a value in the evidence_record block

Run:
    pytest tests/ -v
"""

from pathlib import Path

from pipeline import EVIDENCE_COLUMNS
from schemas import PipelineResult


class TestPipelineResultConstruction:

    def test_minimal_construction_does_not_raise(self):
        """PipelineResult should build cleanly when only Literal fields are supplied."""
        result = PipelineResult(
            final_prediction=0,
            subtype="na",
            confidence="low",
            synthesis_mode="consensus_negative",
        )
        assert result.final_prediction == 0
        assert result.subtype == "na"
        assert result.confidence == "low"
        assert result.synthesis_mode == "consensus_negative"
        # Confirm defaulted fields have their expected types
        assert isinstance(result.contributing_agents, list)
        assert isinstance(result.agent_errors, list)
        assert result.latency_ms == 0.0
        assert result.llm_calls == 0


class TestEvidenceColumnsContract:

    def test_all_evidence_columns_assigned_in_pipeline_source(self):
        """
        Every name in EVIDENCE_COLUMNS must appear as a quoted string key in
        pipeline.py's evidence_record construction block.
        No real pipeline run needed — pure source-level string search.
        """
        src = (Path(__file__).parent.parent / "pipeline.py").read_text(encoding="utf-8")
        missing = [col for col in EVIDENCE_COLUMNS if f'"{col}"' not in src]
        assert not missing, (
            f"These EVIDENCE_COLUMNS entries have no matching key assignment in pipeline.py: {missing}"
        )
