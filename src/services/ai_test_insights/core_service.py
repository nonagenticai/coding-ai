"""AI analysis core service module for AI Test Insights Service.

This module contains the main AITestInsightsService class and core AI agent
initialization logic for PydanticAI integration.
"""

import logging
import uuid
from typing import TYPE_CHECKING, Any

from models.testing_models import QualityResult, TestResult
from services.ai_test_insights.config import (
    DEFAULT_CONFIG,
    AITestInsightsConfig,
    TestInsightType,
    clone_config,
)
from services.ai_test_insights.insight_generation import TestInsightGenerator
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestInsightAnalysis,
    TestInsightRecommendation,
    TestPatternInsight,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)
from services.ai_test_insights.pattern_recognition import TestPatternDetector
from services.ai_test_insights.report_formatter import TestReportFormatter

if TYPE_CHECKING:
    from pydantic_ai import Agent, RunContext

# Try to import pydantic_ai, but make it optional
try:
    from pydantic_ai import Agent, RunContext

    HAS_PYDANTIC_AI = True
except ImportError:
    HAS_PYDANTIC_AI = False
    Agent = None  # type: ignore[misc, assignment]  # ty: ignore[invalid-assignment]
    RunContext = None  # type: ignore[misc, assignment]  # ty: ignore[invalid-assignment]

logger = logging.getLogger(__name__)


class AITestInsightsService:
    """AI-powered test insights service using PydanticAI."""

    def __init__(
        self,
        cache_manager: Any | None = None,
        ai_insights_engine=None,
        config: AITestInsightsConfig = DEFAULT_CONFIG,
    ):
        """Initialize the AI test insights service."""
        self.cache_manager = cache_manager
        self.ai_insights_engine = ai_insights_engine
        self.config = clone_config(config)

        # Initialize modular components
        self.insight_generator = TestInsightGenerator(config)
        self.pattern_detector = TestPatternDetector(config)
        self.report_formatter = TestReportFormatter(config, cache_manager)

        # Initialize PydanticAI agents
        self._init_ai_agents()

        logger.info("AI Test Insights Service initialized with modular components")

    def _init_ai_agents(self) -> None:
        """Initialize PydanticAI agents for different types of analysis."""

        if not HAS_PYDANTIC_AI or not self.config.enable_ai_features:
            logger.warning(
                "pydantic_ai not available or AI features disabled - AI agents will be disabled"
            )
            self.performance_agent = None
            self.reliability_agent = None
            self.coverage_agent = None
            self.quality_agent = None
            self.pattern_agent = None
            self.recommendation_agent = None
            return

        try:
            # Performance analysis agent
            self.performance_agent = Agent(
                model="test",
                output_type=TestPerformanceMetrics,
                instructions="""
                You are an expert performance analyst for software testing.
                Analyze test execution data to identify performance bottlenecks,
                slow tests, and resource usage patterns.
                """,
            )

            # Reliability analysis agent
            self.reliability_agent = Agent(
                model="test",
                output_type=TestReliabilityMetrics,
                instructions="""
                You are an expert reliability analyst for software testing.
                Analyze test execution results to identify reliability issues,
                flaky tests, and failure patterns.
                """,
            )

            # Coverage analysis agent
            self.coverage_agent = Agent(
                model="test",
                output_type=TestCoverageInsights,
                instructions="""
                You are an expert test coverage analyst.
                Analyze test coverage data to identify gaps, redundancies,
                and optimization opportunities.
                """,
            )

            # Quality analysis agent
            self.quality_agent = Agent(
                model="test",
                output_type=TestQualityInsights,
                instructions="""
                You are an expert code quality analyst for testing.
                Analyze code quality metrics and test quality to identify
                improvement opportunities.
                """,
            )

            # Pattern detection agent
            self.pattern_agent = Agent(
                model="test",
                output_type=list[TestPatternInsight],
                instructions="""
                You are an expert pattern recognition analyst for software testing.
                Analyze test execution data to identify patterns, anti-patterns,
                and recurring issues.
                """,
            )

            # Recommendation agent
            self.recommendation_agent = Agent(
                model="test",
                output_type=list[TestInsightRecommendation],
                instructions="""
                You are an expert testing consultant providing actionable
                recommendations for test improvement.
                """,
            )

            logger.info("Successfully initialized all AI agents")

        except Exception as e:
            logger.warning(f"Failed to initialize AI agents: {e}. AI features will be disabled.")
            self.performance_agent = None
            self.reliability_agent = None
            self.coverage_agent = None
            self.quality_agent = None
            self.pattern_agent = None
            self.recommendation_agent = None

    async def _analyze_with_llm(
        self,
        content: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> dict[str, Any] | None:
        """Analyze content using ML backend LLM."""
        if not self.config.use_ml_backend or not self.config.enable_ai_features:
            return None

        try:
            # TODO: Import from coding-ai LLM client when available
            logger.debug("LLM client not available in microservice yet")
            return None
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return None

    async def analyze_test_insights(
        self,
        test_results: list[TestResult],
        quality_results: list[QualityResult],
        time_period: str = "recent",
        include_predictions: bool = True,
    ) -> TestInsightAnalysis:
        """Analyze test insights using modular components."""
        logger.info("Starting AI test insights analysis")

        try:
            analysis = await self.report_formatter.create_comprehensive_analysis(
                test_results=test_results,
                quality_results=quality_results,
                time_period=time_period,
                include_predictions=include_predictions,
            )

            logger.info("AI test insights analysis completed successfully")

            return analysis

        except Exception as e:
            logger.error(f"AI test insights analysis failed: {e}")
            raise

    async def comprehensive_analysis(
        self,
        test_results: list[TestResult],
        quality_results: list[QualityResult],
        code_changes: list[str] | None = None,
        time_period: str = "recent",
        include_predictions: bool = True,
        specialized_analyzers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive analysis using all specialized analyzers."""
        logger.info("Starting comprehensive test analysis")

        try:
            comprehensive_report = await self.report_formatter.generate_comprehensive_report(
                test_results=test_results,
                quality_results=quality_results,
                code_changes=code_changes,
                time_period=time_period,
                include_predictions=include_predictions,
                specialized_analyzers=specialized_analyzers,
            )

            logger.info("Comprehensive analysis completed successfully")

            return comprehensive_report

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise

    async def predict_test_failures(
        self, test_results: list[TestResult], code_changes: list[str] | None = None
    ) -> dict[str, float]:
        """Predict likelihood of test failures based on historical data."""
        predictions = {}

        # Group results by test/language
        test_groups: dict[str, list[TestResult]] = {}
        for result in test_results:
            key = f"{result.language}_{result.test_type or 'unknown'}"
            if key not in test_groups:
                test_groups[key] = []
            test_groups[key].append(result)

        # Calculate failure probability for each group
        for key, group_results in test_groups.items():
            success_rate = sum(1 for r in group_results if r.success) / len(group_results)
            failure_probability = 1 - success_rate

            # Adjust based on recent trends
            recent_results = group_results[-5:]  # Last 5 results
            if recent_results:
                recent_success_rate = sum(1 for r in recent_results if r.success) / len(
                    recent_results
                )
                failure_probability = 0.7 * failure_probability + 0.3 * (1 - recent_success_rate)

            predictions[key] = min(1.0, max(0.0, failure_probability))

        return predictions

    async def generate_test_recommendations(
        self, analysis: TestInsightAnalysis, focus_areas: list[str] | None = None
    ) -> list[TestInsightRecommendation]:
        """Generate additional test recommendations based on analysis."""
        additional_recommendations = []

        # Generate recommendations based on health score
        if analysis.overall_health_score < 70:
            additional_recommendations.append(
                TestInsightRecommendation(
                    recommendation_id=f"health_{str(uuid.uuid4())[:8]}",
                    type=TestInsightType.RECOMMENDATIONS,
                    priority="high",
                    confidence=0.85,
                    title="Comprehensive Test Suite Overhaul",
                    description="Test suite health score is below acceptable threshold",
                    rationale=f"Health score of {analysis.overall_health_score:.1f} indicates multiple issues",
                    implementation_steps=[
                        "Conduct comprehensive test suite audit",
                        "Prioritize high-impact improvements",
                        "Implement systematic test improvements",
                        "Establish ongoing monitoring",
                    ],
                    estimated_effort="high",
                    expected_impact="Improve overall test health by 40-60%",
                    supporting_metrics={
                        "health_score": analysis.overall_health_score,
                        "improvement_potential": analysis.improvement_potential,
                    },
                    affected_components=["entire_test_suite"],
                )
            )

        # Generate recommendations based on patterns
        high_impact_patterns = [p for p in analysis.detected_patterns if p.impact_score > 7]
        if high_impact_patterns:
            additional_recommendations.append(
                TestInsightRecommendation(
                    recommendation_id=f"pattern_{str(uuid.uuid4())[:8]}",
                    type=TestInsightType.PATTERNS,
                    priority="high",
                    confidence=0.8,
                    title="Address High-Impact Test Patterns",
                    description="High-impact patterns detected that require immediate attention",
                    rationale=f"Found {len(high_impact_patterns)} high-impact patterns",
                    implementation_steps=[
                        "Analyze high-impact patterns in detail",
                        "Develop targeted improvement plans",
                        "Implement pattern-specific solutions",
                        "Monitor pattern resolution",
                    ],
                    estimated_effort="medium",
                    expected_impact="Resolve major test suite issues",
                    supporting_metrics={"high_impact_patterns": len(high_impact_patterns)},
                    affected_components=["test_patterns", "test_reliability"],
                )
            )

        return additional_recommendations

    async def get_cached_analysis(self, analysis_id: str) -> TestInsightAnalysis | None:
        """Get cached analysis results."""
        return await self.report_formatter.get_cached_analysis(analysis_id)

    async def get_cached_comprehensive_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        """Get cached comprehensive analysis results."""
        return await self.report_formatter.get_cached_comprehensive_analysis(analysis_id)

    def get_config(self) -> AITestInsightsConfig:
        """Get the current configuration."""
        return self.config

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        # Reinitialize components if configuration changed
        self.insight_generator = TestInsightGenerator(self.config)
        self.pattern_detector = TestPatternDetector(self.config)
        self.report_formatter = TestReportFormatter(self.config, self.cache_manager)
        self._init_ai_agents()


# Factory function for dependency injection
def create_ai_test_insights_service(
    cache_manager: Any | None = None,
    ai_insights_engine=None,
    config: AITestInsightsConfig = DEFAULT_CONFIG,
) -> AITestInsightsService:
    """Create AI test insights service."""
    return AITestInsightsService(
        cache_manager=cache_manager,
        ai_insights_engine=ai_insights_engine,
        config=config,
    )
