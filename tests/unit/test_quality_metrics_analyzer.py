"""Tests for QualityMetricsAnalyzer stub models."""

from __future__ import annotations

import pytest

from services.quality_metrics_analyzer import (
    QualityAnalysisResult,
    QualityRecommendation,
)


class TestQualityRecommendation:
    def test_defaults(self):
        rec = QualityRecommendation(
            recommendation_id="r1",
            title="Improve coverage",
            description="Add more tests",
        )
        assert rec.priority == "medium"
        assert rec.confidence == 0.8
        assert rec.expected_improvement == 10.0
        assert rec.effort_estimate == "medium"
        assert rec.implementation_steps == []
        assert rec.prerequisites == []
        assert rec.tools_required == []
        assert rec.supporting_evidence == {}
        assert rec.affected_components == []
        assert rec.risks == []

    def test_full_fields(self):
        rec = QualityRecommendation(
            recommendation_id="r2",
            title="Fix linting",
            description="Run linter and fix issues",
            priority="high",
            confidence=0.95,
            expected_improvement=25.0,
            implementation_steps=["Step 1", "Step 2"],
            effort_estimate="low",
            prerequisites=["Install ruff"],
            tools_required=["ruff"],
            supporting_evidence={"issues": 42},
            affected_components=["src/main.py"],
            risks=["May break formatting"],
        )
        assert rec.priority == "high"
        assert rec.confidence == 0.95
        assert len(rec.implementation_steps) == 2
        assert rec.tools_required == ["ruff"]

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            QualityRecommendation(
                recommendation_id="r3",
                title="Bad confidence",
                description="Invalid",
                confidence=1.5,
            )

    def test_confidence_lower_bound(self):
        with pytest.raises(Exception):
            QualityRecommendation(
                recommendation_id="r4",
                title="Bad confidence",
                description="Invalid",
                confidence=-0.1,
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            QualityRecommendation(
                recommendation_id="r5",
                title="Bad",
                description="Extra fields",
                not_a_field="oops",
            )


class TestQualityAnalysisResult:
    def test_defaults(self):
        result = QualityAnalysisResult(analysis_id="a1")
        assert result.overall_quality_score == 0.0
        assert result.recommendations == []

    def test_with_recommendations(self):
        rec = QualityRecommendation(
            recommendation_id="r1",
            title="Fix it",
            description="Fix the thing",
        )
        result = QualityAnalysisResult(
            analysis_id="a2",
            overall_quality_score=85.0,
            recommendations=[rec],
        )
        assert result.overall_quality_score == 85.0
        assert len(result.recommendations) == 1
        assert result.recommendations[0].recommendation_id == "r1"

    def test_score_bounds(self):
        with pytest.raises(Exception):
            QualityAnalysisResult(
                analysis_id="a3",
                overall_quality_score=101.0,
            )

    def test_score_lower_bound(self):
        with pytest.raises(Exception):
            QualityAnalysisResult(
                analysis_id="a4",
                overall_quality_score=-1.0,
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            QualityAnalysisResult(
                analysis_id="a5",
                unknown_field="bad",
            )

    def test_multiple_recommendations(self):
        recs = [
            QualityRecommendation(
                recommendation_id=f"r{i}",
                title=f"Rec {i}",
                description=f"Description {i}",
                priority="high" if i % 2 == 0 else "low",
            )
            for i in range(5)
        ]
        result = QualityAnalysisResult(
            analysis_id="a6",
            overall_quality_score=50.0,
            recommendations=recs,
        )
        assert len(result.recommendations) == 5
        assert result.recommendations[0].priority == "high"
        assert result.recommendations[1].priority == "low"
