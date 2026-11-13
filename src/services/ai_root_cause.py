"""
AI-powered root cause analysis engine for test failures.

This module provides specialized AI-powered analysis for identifying the root causes
of test failures with high accuracy and detailed reasoning. It uses advanced pattern
matching, contextual analysis, and machine learning to understand complex failure
scenarios.

The engine analyzes:
- Code structure and logic flows
- Data types and validation issues
- Environment and configuration problems
- Dependency and integration failures
- Performance and resource constraints
- Concurrency and timing issues
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from coding_client.llm import (
    last_failover_chain_id,
    last_inference_id,
    llm_request_context,
)

from models.ai_test_failure_models import (
    ConfidenceLevel,
    ErrorDetails,
    RootCause,
    SeverityLevel,
    TestFailureData,
)
from services.llm_agents import SCHEMA_TEST_FAILURE, make_agent_runner

logger = logging.getLogger(__name__)


class AgentAnalysisError(Exception):
    """Raised when an LLM-backed root-cause dimension fails to run.

    Signals that a sub-analysis could not obtain a result from the LLM plane
    (agent unreachable / provider error / malformed response). This is distinct
    from a sub-analysis that ran successfully and simply found no root causes:
    the former is a degraded/outage condition, the latter is a genuine empty
    result. :meth:`AIRootCauseEngine.analyze_root_causes` counts these so it can
    tell "every agent failed" (outage) apart from "all agents genuinely empty".
    """


@dataclass
class AIAnalysisContext:
    """Context information for root cause analysis."""

    failure_data: TestFailureData
    error_details: ErrorDetails
    code_snippets: list[str] | None = None
    environment_info: dict[str, Any] | None = None
    historical_failures: list[str] | None = None
    dependency_graph: dict[str, list[str]] | None = None


class AIRootCauseEngine:
    """
    AI-powered root cause analysis engine.

    This engine specializes in identifying the underlying causes of test failures
    by analyzing multiple dimensions of the failure context:

    1. Syntactic Analysis - Code structure and syntax issues
    2. Semantic Analysis - Logic and data flow problems
    3. Environmental Analysis - Configuration and setup issues
    4. Integration Analysis - Dependency and service interactions
    5. Performance Analysis - Resource and timing constraints
    6. Historical Analysis - Pattern recognition from past failures
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the AI root cause analysis engine.

        Args:
            model_name: Retained for backwards compatibility; ignored. Provider
                selection is server-side in inference-router.
        """
        self.model_name = model_name or SCHEMA_TEST_FAILURE
        self.logger = logging.getLogger(__name__)

        # Initialize specialized AI agents
        self._init_analysis_agents()

        # Knowledge base for common patterns and solutions
        self._knowledge_base = self._initialize_knowledge_base()

        # Cache for analysis results
        self._analysis_cache: dict[str, list[RootCause]] = {}

    def _init_analysis_agents(self) -> None:
        """Initialize specialized AI agents for different analysis types."""

        # Syntactic analysis agent
        self.syntactic_agent = make_agent_runner(
            worker_name="cai-root-cause-syntactic",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[RootCause],
            system_prompt="""
            You are an expert in code syntax analysis and compilation errors.
            Your task is to identify syntax-related root causes of test failures.

            Focus on:
            - Syntax errors and parsing issues
            - Import and module resolution problems
            - Type annotation and declaration issues
            - Indentation and formatting problems
            - Missing or incorrect punctuation
            - Language-specific syntax violations

            For each root cause:
            1. Provide specific line/column information when available
            2. Explain the exact syntax violation
            3. Suggest the correct syntax
            4. Assess the severity and confidence
            """,
        )

        # Semantic analysis agent
        self.semantic_agent = make_agent_runner(
            worker_name="cai-root-cause-semantic",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[RootCause],
            system_prompt="""
            You are an expert in program logic and semantic analysis.
            Your task is to identify logic-related root causes of test failures.

            Focus on:
            - Logic errors and incorrect algorithms
            - Data type mismatches and conversions
            - Null/undefined value handling
            - Boundary conditions and edge cases
            - State management and variable scope
            - Control flow and branching issues

            For each root cause:
            1. Explain the logical flaw or incorrect assumption
            2. Describe the expected vs actual behavior
            3. Identify the specific code location if possible
            4. Suggest the correct logic or approach
            """,
        )

        # Environmental analysis agent
        self.environmental_agent = make_agent_runner(
            worker_name="cai-root-cause-environmental",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[RootCause],
            system_prompt="""
            You are an expert in software environment and configuration analysis.
            Your task is to identify environment-related root causes of test failures.

            Focus on:
            - Configuration file issues
            - Environment variable problems
            - Path and file system issues
            - Permission and access problems
            - Version compatibility issues
            - System resource constraints

            For each root cause:
            1. Identify the specific configuration or environment issue
            2. Explain how it affects the test execution
            3. Suggest the correct configuration or setup
            4. Consider platform-specific differences
            """,
        )

        # Integration analysis agent
        self.integration_agent = make_agent_runner(
            worker_name="cai-root-cause-integration",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[RootCause],
            system_prompt="""
            You are an expert in system integration and dependency analysis.
            Your task is to identify integration-related root causes of test failures.

            Focus on:
            - Dependency version conflicts
            - Service communication failures
            - API contract violations
            - Database connection issues
            - Network connectivity problems
            - Third-party service failures

            For each root cause:
            1. Identify the specific integration point that failed
            2. Explain the expected vs actual behavior
            3. Consider version compatibility and API changes
            4. Suggest troubleshooting and resolution steps
            """,
        )

        # Performance analysis agent
        self.performance_agent = make_agent_runner(
            worker_name="cai-root-cause-performance",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[RootCause],
            system_prompt="""
            You are an expert in performance analysis and resource management.
            Your task is to identify performance-related root causes of test failures.

            Focus on:
            - Memory leaks and out-of-memory errors
            - CPU usage and processing bottlenecks
            - I/O operations and disk space issues
            - Network latency and bandwidth problems
            - Timeout and response time issues
            - Concurrency and thread safety problems

            For each root cause:
            1. Identify the specific resource constraint or bottleneck
            2. Explain how it leads to test failure
            3. Suggest optimization strategies
            4. Consider scalability and load factors
            """,
        )

    def _initialize_knowledge_base(self) -> dict[str, Any]:
        """Initialize the knowledge base with common patterns and solutions."""
        return {
            "common_patterns": {
                "assertion_failures": {
                    "none_instead_of_value": {
                        "pattern": r"expected.*actual.*None",
                        "cause": "null_reference",
                        "confidence": 0.8,
                    },
                    "type_mismatch": {
                        "pattern": r"expected.*<class.*actual.*<class",
                        "cause": "type_mismatch",
                        "confidence": 0.7,
                    },
                    "off_by_one": {
                        "pattern": r"expected.*\d+.*actual.*\d+",
                        "cause": "logic_error",
                        "confidence": 0.6,
                    },
                },
                "import_errors": {
                    "module_not_found": {
                        "pattern": r"ModuleNotFoundError.*No module named",
                        "cause": "dependency_missing",
                        "confidence": 0.9,
                    },
                    "circular_import": {
                        "pattern": r"cannot import name.*from partially initialized module",
                        "cause": "dependency_missing",
                        "confidence": 0.9,
                    },
                },
                "runtime_errors": {
                    "attribute_error": {
                        "pattern": r"AttributeError.*object has no attribute",
                        "cause": "null_reference",
                        "confidence": 0.8,
                    },
                    "key_error": {
                        "pattern": r"KeyError.*",
                        "cause": "data_validation",
                        "confidence": 0.7,
                    },
                },
            },
            "severity_mappings": {
                "syntax_error": SeverityLevel.CRITICAL,
                "import_error": SeverityLevel.HIGH,
                "assertion_error": SeverityLevel.MEDIUM,
                "runtime_error": SeverityLevel.HIGH,
                "timeout_error": SeverityLevel.MEDIUM,
                "memory_error": SeverityLevel.HIGH,
            },
            "confidence_factors": {
                "exact_match": 0.95,
                "pattern_match": 0.8,
                "contextual_match": 0.6,
                "heuristic_match": 0.4,
                "fallback": 0.2,
            },
        }

    async def analyze_root_causes(
        self,
        context: AIAnalysisContext,
        max_causes: int = 5,
        minimum_confidence: float = 0.1,
    ) -> list[RootCause]:
        """
        Perform comprehensive root cause analysis.

        Args:
            context: Analysis context with failure data and additional information
            max_causes: Maximum number of root causes to identify
            minimum_confidence: Minimum confidence threshold for results

        Returns:
            List of identified root causes ranked by confidence
        """
        start_time = datetime.now(UTC)

        try:
            self.logger.info(
                f"Starting root cause analysis for failure {context.failure_data.failure_id}"
            )

            # Check cache first
            cache_key = self._create_cache_key(context)
            if cache_key in self._analysis_cache:
                self.logger.info("Using cached analysis results")
                return self._analysis_cache[cache_key]

            # Perform multi-dimensional analysis under the registry schema_id.
            #
            # Each LLM-backed dimension is attempted independently and its
            # success/failure recorded. A dimension that cannot reach the LLM
            # plane raises AgentAnalysisError; we count those so that a total
            # outage (every agent failed) is distinguishable from a genuine
            # "the LLM ran and found nothing" result. Without this, each
            # dimension silently swallowing its error to `[]` made a full LLM
            # outage indistinguishable from success-with-no-causes, and the
            # degraded fallback below was unreachable dead code.
            all_causes: list[RootCause] = []
            llm_dimensions = (
                self._analyze_syntactic_issues,
                self._analyze_semantic_issues,
                self._analyze_environmental_issues,
                self._analyze_integration_issues,
                self._analyze_performance_issues,
            )
            llm_attempts = len(llm_dimensions)
            llm_failures = 0
            pattern_causes: list[RootCause] = []

            with llm_request_context(schema_id=SCHEMA_TEST_FAILURE):
                for analyze_dimension in llm_dimensions:
                    try:
                        all_causes.extend(await analyze_dimension(context))
                    except AgentAnalysisError as agent_error:
                        llm_failures += 1
                        self.logger.error(
                            "Root-cause dimension failed (LLM plane): %s",
                            str(agent_error),
                        )

                # Pattern-based analysis performs no LLM call — it is
                # best-effort regex/knowledge-base matching and never counts
                # toward the outage tally.
                pattern_causes = await self._analyze_known_patterns(context)
                all_causes.extend(pattern_causes)

            # Total LLM outage: every AI agent failed. This is NOT a genuine
            # "no root causes found" result — the LLM plane was unreachable —
            # so surface an explicit degraded fallback instead of a silent
            # empty success. The degraded result is deliberately NOT cached, so
            # a later request re-runs the full analysis once the LLM recovers.
            if llm_attempts and llm_failures == llm_attempts:
                self.logger.warning(
                    "All %d LLM root-cause agents failed (LLM plane unreachable); "
                    "returning degraded fallback analysis for failure %s",
                    llm_attempts,
                    context.failure_data.failure_id,
                )
                degraded_causes = self._create_fallback_analysis(context)
                # Preserve any real, non-LLM pattern matches alongside the
                # degraded marker so heuristic signal is not lost.
                degraded_causes.extend(pattern_causes)
                return degraded_causes

            # Consolidate and rank causes
            consolidated_causes = self._consolidate_causes(all_causes)

            # Filter by confidence and limit results
            filtered_causes = [
                cause
                for cause in consolidated_causes
                if cause.confidence_score >= minimum_confidence
            ][:max_causes]

            # Cache results
            self._analysis_cache[cache_key] = filtered_causes

            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            self.logger.info(
                "Root cause analysis completed in %.2fs, identified %d causes "
                "(inference_id=%s, failover_chain_id=%s)",
                processing_time,
                len(filtered_causes),
                last_inference_id(),
                last_failover_chain_id(),
            )

            return filtered_causes

        except Exception as e:
            self.logger.error(f"Root cause analysis failed: {str(e)}")

            # Return fallback analysis
            return self._create_fallback_analysis(context)

    async def _analyze_syntactic_issues(self, context: AIAnalysisContext) -> list[RootCause]:
        """Analyze syntax-related issues in the test failure."""
        try:
            analysis_input = self._prepare_syntactic_input(context)

            result = await self.syntactic_agent.run(
                f"Analyze syntax issues in this test failure:\n\n{analysis_input}"
            )

            # Validate and enhance results
            enhanced_causes = []
            for cause in result.output:
                enhanced_cause = self._enhance_cause_with_context(cause, context, "syntactic")
                enhanced_causes.append(enhanced_cause)

            return enhanced_causes

        except Exception as e:
            self.logger.error(f"Syntactic analysis failed: {str(e)}")
            raise AgentAnalysisError(f"syntactic analysis failed: {e}") from e

    async def _analyze_semantic_issues(self, context: AIAnalysisContext) -> list[RootCause]:
        """Analyze semantic/logic issues in the test failure."""
        try:
            analysis_input = self._prepare_semantic_input(context)

            result = await self.semantic_agent.run(
                f"Analyze semantic/logic issues in this test failure:\n\n{analysis_input}"
            )

            # Validate and enhance results
            enhanced_causes = []
            for cause in result.output:
                enhanced_cause = self._enhance_cause_with_context(cause, context, "semantic")
                enhanced_causes.append(enhanced_cause)

            return enhanced_causes

        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {str(e)}")
            raise AgentAnalysisError(f"semantic analysis failed: {e}") from e

    async def _analyze_environmental_issues(self, context: AIAnalysisContext) -> list[RootCause]:
        """Analyze environment-related issues in the test failure."""
        try:
            analysis_input = self._prepare_environmental_input(context)

            result = await self.environmental_agent.run(
                f"Analyze environmental issues in this test failure:\n\n{analysis_input}"
            )

            # Validate and enhance results
            enhanced_causes = []
            for cause in result.output:
                enhanced_cause = self._enhance_cause_with_context(cause, context, "environmental")
                enhanced_causes.append(enhanced_cause)

            return enhanced_causes

        except Exception as e:
            self.logger.error(f"Environmental analysis failed: {str(e)}")
            raise AgentAnalysisError(f"environmental analysis failed: {e}") from e

    async def _analyze_integration_issues(self, context: AIAnalysisContext) -> list[RootCause]:
        """Analyze integration-related issues in the test failure."""
        try:
            analysis_input = self._prepare_integration_input(context)

            result = await self.integration_agent.run(
                f"Analyze integration issues in this test failure:\n\n{analysis_input}"
            )

            # Validate and enhance results
            enhanced_causes = []
            for cause in result.output:
                enhanced_cause = self._enhance_cause_with_context(cause, context, "integration")
                enhanced_causes.append(enhanced_cause)

            return enhanced_causes

        except Exception as e:
            self.logger.error(f"Integration analysis failed: {str(e)}")
            raise AgentAnalysisError(f"integration analysis failed: {e}") from e

    async def _analyze_performance_issues(self, context: AIAnalysisContext) -> list[RootCause]:
        """Analyze performance-related issues in the test failure."""
        try:
            analysis_input = self._prepare_performance_input(context)

            result = await self.performance_agent.run(
                f"Analyze performance issues in this test failure:\n\n{analysis_input}"
            )

            # Validate and enhance results
            enhanced_causes = []
            for cause in result.output:
                enhanced_cause = self._enhance_cause_with_context(cause, context, "performance")
                enhanced_causes.append(enhanced_cause)

            return enhanced_causes

        except Exception as e:
            self.logger.error(f"Performance analysis failed: {str(e)}")
            raise AgentAnalysisError(f"performance analysis failed: {e}") from e

    async def _analyze_known_patterns(self, context: AIAnalysisContext) -> list[RootCause]:
        """Analyze failure against known patterns in the knowledge base."""
        causes = []

        try:
            raw_output = context.failure_data.raw_output

            # Check against known patterns
            patterns = self._knowledge_base.get("common_patterns", {})

            for category, category_patterns in patterns.items():
                for pattern_name, pattern_info in category_patterns.items():
                    if self._matches_pattern(raw_output, pattern_info.get("pattern", "")):
                        cause = self._create_pattern_based_cause(
                            pattern_name, pattern_info, context
                        )
                        causes.append(cause)

            return causes

        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {str(e)}")
            return []

    def _prepare_syntactic_input(self, context: AIAnalysisContext) -> str:
        """Prepare input for syntactic analysis."""
        parts = [
            f"Error Type: {str(context.error_details.error_type)}",
            f"Error Message: {context.error_details.error_message}",
        ]

        if context.error_details.file_path:
            parts.append(f"File: {context.error_details.file_path}")

        if context.error_details.line_number:
            parts.append(f"Line: {context.error_details.line_number}")

        if context.error_details.traceback:
            parts.append(f"Traceback:\n{context.error_details.traceback}")

        if context.code_snippets:
            parts.append("Code Snippets:")
            parts.extend(context.code_snippets)

        return "\n".join(parts)

    def _prepare_semantic_input(self, context: AIAnalysisContext) -> str:
        """Prepare input for semantic analysis."""
        parts = [
            f"Test Name: {context.failure_data.context.test_name}",
            f"Error Type: {str(context.error_details.error_type)}",
            f"Error Message: {context.error_details.error_message}",
        ]

        if context.error_details.stack_trace:
            parts.append("Stack Trace:")
            parts.extend(context.error_details.stack_trace)

        parts.append(f"Raw Output:\n{context.failure_data.raw_output}")

        return "\n".join(parts)

    def _prepare_environmental_input(self, context: AIAnalysisContext) -> str:
        """Prepare input for environmental analysis."""
        parts = [
            f"Error Type: {str(context.error_details.error_type)}",
            f"Error Message: {context.error_details.error_message}",
            f"Framework: {context.failure_data.context.framework!s}",
        ]

        if context.environment_info:
            parts.append(f"Environment Info: {json.dumps(context.environment_info, indent=2)}")

        if context.failure_data.context.execution_environment:
            parts.append(
                f"Execution Environment: {json.dumps(context.failure_data.context.execution_environment, indent=2)}"
            )

        return "\n".join(parts)

    def _prepare_integration_input(self, context: AIAnalysisContext) -> str:
        """Prepare input for integration analysis."""
        parts = [
            f"Error Type: {str(context.error_details.error_type)}",
            f"Error Message: {context.error_details.error_message}",
        ]

        if context.dependency_graph:
            parts.append(f"Dependency Graph: {json.dumps(context.dependency_graph, indent=2)}")

        if context.error_details.traceback:
            parts.append(f"Traceback:\n{context.error_details.traceback}")

        return "\n".join(parts)

    def _prepare_performance_input(self, context: AIAnalysisContext) -> str:
        """Prepare input for performance analysis."""
        parts = [
            f"Error Type: {str(context.error_details.error_type)}",
            f"Error Message: {context.error_details.error_message}",
        ]

        if context.failure_data.execution_time:
            parts.append(f"Execution Time: {context.failure_data.execution_time}s")

        if context.failure_data.memory_usage:
            parts.append(f"Memory Usage: {context.failure_data.memory_usage} bytes")

        return "\n".join(parts)

    def _enhance_cause_with_context(
        self, cause: RootCause, context: AIAnalysisContext, analysis_type: str
    ) -> RootCause:
        """Enhance a root cause with additional context and validation."""
        # Ensure unique ID
        cause.cause_id = f"{analysis_type}_{str(uuid.uuid4())[:8]}"

        # Add analysis type to evidence
        if not cause.evidence:
            cause.evidence = []
        cause.evidence.append(f"Identified through {analysis_type} analysis")

        # Adjust confidence based on analysis type and context
        if analysis_type in ["syntactic", "pattern"]:
            # Syntactic and pattern analysis are typically more reliable
            cause.confidence_score = min(cause.confidence_score * 1.1, 1.0)
        elif analysis_type in ["semantic", "integration"]:
            # Semantic and integration analysis require more interpretation
            cause.confidence_score = cause.confidence_score * 0.9

        # Map confidence score to level
        if cause.confidence_score >= 0.9:
            cause.confidence = ConfidenceLevel.VERY_HIGH
        elif cause.confidence_score >= 0.75:
            cause.confidence = ConfidenceLevel.HIGH
        elif cause.confidence_score >= 0.5:
            cause.confidence = ConfidenceLevel.MEDIUM
        elif cause.confidence_score >= 0.25:
            cause.confidence = ConfidenceLevel.LOW
        else:
            cause.confidence = ConfidenceLevel.VERY_LOW

        return cause

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a pattern."""
        if not pattern:
            return False

        import re

        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except re.error:
            return pattern.lower() in text.lower()

    def _create_pattern_based_cause(
        self,
        pattern_name: str,
        pattern_info: dict[str, Any],
        context: AIAnalysisContext,
    ) -> RootCause:
        """Create a root cause based on a known pattern."""
        cause_category = pattern_info.get("cause", "unknown")
        confidence_score = pattern_info.get("confidence", 0.5)

        # Map to severity
        error_type = str(context.error_details.error_type)
        severity = self._knowledge_base.get("severity_mappings", {}).get(
            error_type, SeverityLevel.MEDIUM
        )

        # Map confidence score to level
        if confidence_score >= 0.9:
            confidence = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.75:
            confidence = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.25:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.VERY_LOW

        return RootCause(
            cause_id=f"pattern_{pattern_name}_{str(uuid.uuid4())[:8]}",
            category=cause_category,
            description=f"Test failure matches known pattern: {pattern_name}",
            severity=severity,
            confidence=confidence,
            confidence_score=confidence_score,
            evidence=[f"Pattern match: {pattern_name}"],
            related_patterns=[pattern_name],
        )

    def _consolidate_causes(self, causes: list[RootCause]) -> list[RootCause]:
        """Consolidate similar causes and rank by confidence."""
        if not causes:
            return []

        # Group similar causes
        consolidated = {}
        for cause in causes:
            # Use category and first 50 chars of description as key
            key = f"{cause.category}_{cause.description[:50]}"

            if key not in consolidated:
                consolidated[key] = cause
            else:
                # Merge evidence and adjust confidence
                existing = consolidated[key]
                existing.evidence.extend(cause.evidence)
                existing.confidence_score = max(existing.confidence_score, cause.confidence_score)

                # Update confidence level based on score
                if existing.confidence_score >= 0.9:
                    existing.confidence = ConfidenceLevel.VERY_HIGH
                elif existing.confidence_score >= 0.75:
                    existing.confidence = ConfidenceLevel.HIGH
                elif existing.confidence_score >= 0.5:
                    existing.confidence = ConfidenceLevel.MEDIUM
                elif existing.confidence_score >= 0.25:
                    existing.confidence = ConfidenceLevel.LOW
                else:
                    existing.confidence = ConfidenceLevel.VERY_LOW

        # Sort by confidence score (highest first)
        sorted_causes = sorted(
            consolidated.values(), key=lambda x: x.confidence_score, reverse=True
        )

        return sorted_causes

    def _create_cache_key(self, context: AIAnalysisContext) -> str:
        """Create a cache key for the analysis context."""
        key_parts = [
            context.failure_data.failure_id,
            str(context.error_details.error_type),
            context.error_details.error_message[:100],
        ]

        return "|".join(key_parts)

    def _create_fallback_analysis(self, context: AIAnalysisContext) -> list[RootCause]:
        """Create fallback analysis when AI analysis fails."""
        fallback_cause = RootCause(
            cause_id=f"fallback_{str(uuid.uuid4())[:8]}",
            category="unknown",
            description=f"Test failed with {str(context.error_details.error_type)}: {context.error_details.error_message}",
            severity=SeverityLevel.MEDIUM,
            confidence=ConfidenceLevel.VERY_LOW,
            confidence_score=0.2,
            evidence=["Fallback analysis due to AI analysis failure"],
        )

        return [fallback_cause]
