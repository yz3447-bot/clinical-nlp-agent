"""
tests/test_synthesizer.py — Unit tests for SynthesizerAgent core logic.

Tests cover:
  - JSON parsing robustness (_parse_llm_json)
  - Consensus early exit (positive + negative)
  - Post-hoc confidence correction (3 rules)
  - Fallback result when LLM fails
  - Subtype validation

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # shorter tracebacks
"""

import pytest
from agents.synthesizer_agent import (
    _parse_llm_json,
    _fallback_result,
    _apply_confidence_correction,
    _compute_score,
    run_synthesizer_agent,
)
from schemas import (
    ClinTextAgentOutput,
    DiagnosisAgentOutput,
    MedicationAgentOutput,
    SynthesizerOutput,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def clin_positive():
    return ClinTextAgentOutput(
        symptoms_found=True,
        confidence="high",
        confidence_score=0.7,
        evidence_list=["progressive memory loss documented", "cognitive decline in A&P"],
        reasoning="Clear dementia documented.",
        assessment="AD/ADRD present",
    )

@pytest.fixture
def clin_negative():
    return ClinTextAgentOutput(
        symptoms_found=False,
        confidence="low",
        confidence_score=0.1,
        evidence_list=[],
        reasoning="No cognitive symptoms found.",
        assessment="No AD/ADRD",
    )

@pytest.fixture
def med_positive():
    return MedicationAgentOutput(
        meds_found=True,
        medications=["donepezil"],
        status="current",
        source="structured",
        confidence="high",
        confidence_score=0.9,
        reasoning="Donepezil prescribed.",
        assessment="Active AD medication",
    )

@pytest.fixture
def med_negative():
    return MedicationAgentOutput(
        meds_found=False,
        medications=[],
        status="none",
        source="none",
        confidence="low",
        confidence_score=0.0,
        reasoning="No AD medications found.",
        assessment="No medication evidence",
    )

@pytest.fixture
def dx_positive():
    return DiagnosisAgentOutput(
        dx_found=True,
        matched_codes=["F0280"],
        is_current_diagnosis=True,
        confidence="high",
        confidence_score=0.8,
        reasoning="F0280 present and current.",
        assessment="Current AD diagnosis",
    )

@pytest.fixture
def dx_negative():
    return DiagnosisAgentOutput(
        dx_found=False,
        matched_codes=[],
        is_current_diagnosis=False,
        confidence="low",
        confidence_score=0.0,
        reasoning="No AD/ADRD ICD codes.",
        assessment="No ICD evidence",
    )


# ── _parse_llm_json ──────────────────────────────────────────────────────────

class TestParseLlmJson:

    def test_clean_json(self):
        raw = '{"final_prediction": 1, "subtype": "ad", "confidence": "high"}'
        result = _parse_llm_json(raw)
        assert result["final_prediction"] == 1
        assert result["subtype"] == "ad"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"final_prediction": 0, "subtype": "na"}\n```'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result["final_prediction"] == 0

    def test_json_with_plain_fences(self):
        raw = '```{"final_prediction": 1, "subtype": "nsd"}```'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result["subtype"] == "nsd"

    def test_json_embedded_in_prose(self):
        raw = 'Here is my analysis:\n{"final_prediction": 1, "subtype": "vd", "confidence": "medium"}\nEnd.'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result["subtype"] == "vd"

    def test_completely_invalid_returns_none(self):
        result = _parse_llm_json("This is not JSON at all.")
        assert result is None

    def test_empty_string_returns_none(self):
        result = _parse_llm_json("")
        assert result is None


# ── _compute_score ───────────────────────────────────────────────────────────

class TestComputeScore:

    def test_all_positive(self, clin_positive, med_positive, dx_positive):
        score, contributing = _compute_score(clin_positive, med_positive, dx_positive)
        assert score == 7  # clin=3, med=3, dx=1
        assert "ClinTextAgent" in contributing
        assert "MedicationAgent" in contributing
        assert "DiagnosisAgent" in contributing

    def test_all_negative(self, clin_negative, med_negative, dx_negative):
        score, contributing = _compute_score(clin_negative, med_negative, dx_negative)
        assert score == 0
        assert contributing == []

    def test_partial(self, clin_positive, med_negative, dx_positive):
        score, contributing = _compute_score(clin_positive, med_negative, dx_positive)
        assert score == 4  # clin=3, dx=1
        assert "MedicationAgent" not in contributing


# ── Consensus early exit ─────────────────────────────────────────────────────

class TestConsensusEarlyExit:

    def test_consensus_positive_skips_llm(self, clin_positive, med_positive, dx_negative):
        """med + clin both positive → predict 1 without LLM call."""
        result = run_synthesizer_agent(dx_negative, med_positive, clin_positive, llm=None)
        assert result.final_prediction == 1
        assert result.synthesis_mode == "consensus_positive"
        assert result.confidence == "high"
        assert result.discrepancy is None

    def test_consensus_negative_skips_llm(self, clin_negative, med_negative, dx_negative):
        """All three negative → predict 0 without LLM call."""
        result = run_synthesizer_agent(dx_negative, med_negative, clin_negative, llm=None)
        assert result.final_prediction == 0
        assert result.synthesis_mode == "consensus_negative"
        assert result.confidence == "high"
        assert result.subtype == "na"

    def test_consensus_positive_returns_required_fields(self, clin_positive, med_positive, dx_negative):
        result = run_synthesizer_agent(dx_negative, med_positive, clin_positive, llm=None)
        # Pydantic validation confirms all fields exist, have correct types, and valid Literal values
        validated = SynthesizerOutput(**result.model_dump())
        assert validated.final_prediction == result.final_prediction


# ── _apply_confidence_correction ────────────────────────────────────────────

class TestConfidenceCorrection:

    def _base_result(self, confidence="high"):
        return {
            "final_prediction": 1,
            "subtype": "ad",
            "confidence": confidence,
            "discrepancy": None,
            "overruled_agents": [],
            "confidence_correction": None,
        }

    def test_rule1_discrepancy_plus_overruled_downgrades_high_to_medium(
        self, clin_positive, med_negative, dx_positive
    ):
        result = self._base_result("high")
        result["discrepancy"] = "ClinTextAgent and MedicationAgent disagreed."
        result["overruled_agents"] = ["MedicationAgent: overruled due to text evidence"]
        out = _apply_confidence_correction(result, clin_positive, med_negative, dx_positive)
        assert out["confidence"] == "medium"
        assert "confidence_correction" in out

    def test_rule1_does_not_downgrade_medium(
        self, clin_positive, med_negative, dx_positive
    ):
        result = self._base_result("medium")
        result["discrepancy"] = "some discrepancy"
        result["overruled_agents"] = ["MedicationAgent: reason"]
        out = _apply_confidence_correction(result, clin_positive, med_negative, dx_positive)
        assert out["confidence"] == "medium"  # already medium, no change

    def test_rule2_weak_evidence_downgrades_to_low(
        self, dx_negative, med_negative
    ):
        clin_weak = ClinTextAgentOutput(
            symptoms_found=True,
            confidence="low",
            confidence_score=0.2,
            evidence_list=["one vague mention"],
        )
        result = self._base_result("high")
        out = _apply_confidence_correction(result, clin_weak, med_negative, dx_negative)
        assert out["confidence"] == "low"

    def test_rule3_sparse_evidence_downgrades_high_to_low(
        self, clin_negative, med_negative, dx_positive
    ):
        clin_sparse = ClinTextAgentOutput(
            symptoms_found=True,
            confidence="low",
            confidence_score=0.2,
            evidence_list=["one item"],  # < 2
        )
        result = self._base_result("high")
        out = _apply_confidence_correction(result, clin_sparse, med_negative, dx_positive)
        assert out["confidence"] == "low"

    def test_no_correction_when_strong_evidence(
        self, clin_positive, med_positive, dx_positive
    ):
        result = self._base_result("high")
        out = _apply_confidence_correction(result, clin_positive, med_positive, dx_positive)
        assert out["confidence"] == "high"
        assert out.get("confidence_correction") is None


# ── _fallback_result ─────────────────────────────────────────────────────────

class TestFallbackResult:

    def test_med_positive_predicts_1_high(self, clin_negative, med_positive, dx_negative):
        result = _fallback_result(3, ["MedicationAgent"], clin_negative, med_positive, dx_negative)
        assert result["final_prediction"] == 1
        assert result["confidence"] == "high"
        assert "Fallback" in result["causal_chain"]

    def test_all_negative_predicts_0(self, clin_negative, med_negative, dx_negative):
        result = _fallback_result(0, [], clin_negative, med_negative, dx_negative)
        assert result["final_prediction"] == 0
        assert result["subtype"] == "na"

    def test_clin_only_predicts_1_low(self, clin_positive, med_negative, dx_negative):
        result = _fallback_result(3, ["ClinTextAgent"], clin_positive, med_negative, dx_negative)
        assert result["final_prediction"] == 1
        assert result["confidence"] == "low"

    def test_clin_and_dx_predicts_1_medium(self, clin_positive, med_negative, dx_positive):
        result = _fallback_result(4, ["ClinTextAgent", "DiagnosisAgent"],
                                  clin_positive, med_negative, dx_positive)
        assert result["final_prediction"] == 1
        assert result["confidence"] == "medium"

    def test_fallback_always_has_required_fields(self, clin_negative, med_negative, dx_negative):
        result = _fallback_result(0, [], clin_negative, med_negative, dx_negative)
        for field in ["final_prediction", "subtype", "confidence",
                      "causal_chain", "contributing_agents", "overruled_agents"]:
            assert field in result


# ── Subtype validation ───────────────────────────────────────────────────────

class TestSubtypeValidation:

    VALID_SUBTYPES = {"ad", "vd", "ftd", "nsd", "na"}

    def test_consensus_positive_returns_valid_subtype(
        self, clin_positive, med_positive, dx_negative
    ):
        result = run_synthesizer_agent(dx_negative, med_positive, clin_positive, llm=None)
        assert result.subtype in self.VALID_SUBTYPES

    def test_consensus_negative_returns_na(
        self, clin_negative, med_negative, dx_negative
    ):
        result = run_synthesizer_agent(dx_negative, med_negative, clin_negative, llm=None)
        assert result.subtype == "na"
