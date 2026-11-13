"""
AI-powered intelligent fix suggestion system for test failures.

This module provides contextual, actionable fix suggestions based on comprehensive
analysis of test failures. It generates specific, prioritized recommendations
with implementation details, code examples, and validation steps.

The system considers:
- Root cause analysis results
- Code context and patterns
- Best practices and conventions
- Historical fix success rates
- Implementation complexity
- Potential side effects
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
    FixSuggestion,
    RootCause,
    TestFailureData,
)
from services.llm_agents import SCHEMA_TEST_FAILURE, make_agent_runner

logger = logging.getLogger(__name__)


@dataclass
class FixContext:
    """Context information for fix suggestion generation."""

    failure_data: TestFailureData
    error_details: ErrorDetails
    root_causes: list[RootCause]
    code_context: dict[str, str] | None = None
    project_structure: dict[str, Any] | None = None
    historical_fixes: list[dict[str, Any]] | None = None
    team_preferences: dict[str, Any] | None = None


class AIFixSuggestionEngine:
    """
    AI-powered fix suggestion engine.

    This engine generates intelligent, actionable fix suggestions by analyzing:

    1. Root Cause Mapping - Direct solutions for identified causes
    2. Code Context Analysis - Understanding the surrounding code
    3. Pattern-Based Suggestions - Solutions based on common patterns
    4. Best Practice Recommendations - Industry standard approaches
    5. Progressive Complexity - From simple to complex solutions
    6. Risk Assessment - Evaluating potential side effects
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the AI fix suggestion engine.

        Args:
            model_name: Retained for backwards compatibility; ignored.
                Provider selection is server-side in inference-router.
        """
        self.model_name = model_name or SCHEMA_TEST_FAILURE
        self.logger = logging.getLogger(__name__)

        # Initialize specialized AI agents
        self._init_suggestion_agents()

        # Solution knowledge base
        self._solution_library = self._initialize_solution_library()

        # Fix success tracking
        self._fix_success_rates: dict[str, float] = {}

        # Suggestion cache
        self._suggestion_cache: dict[str, list[FixSuggestion]] = {}

    def _init_suggestion_agents(self) -> None:
        """Initialize specialized AI agents for different suggestion types."""

        # Immediate fix agent
        self.immediate_fix_agent = make_agent_runner(
            worker_name="cai-fix-suggestion-immediate-fix",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[FixSuggestion],
            system_prompt="""
            You are an expert software engineer who specializes in providing immediate,
            actionable solutions to test failures. Your task is to generate specific,
            implementable fixes that directly address the root causes.

            For each suggestion:
            1. Provide the exact code changes needed
            2. Explain why this fix addresses the root cause
            3. Include specific line numbers when possible
            4. Estimate implementation effort accurately
            5. Consider potential side effects

            Focus on:
            - Direct, minimal changes that resolve the issue
            - Clear, step-by-step instructions
            - Concrete code examples
            - Immediate implementability

            Prioritize solutions by:
            - Directness of impact on the root cause
            - Simplicity of implementation
            - Minimal risk of side effects
            - Alignment with best practices
            """,
        )

        # Preventive fix agent
        self.preventive_fix_agent = make_agent_runner(
            worker_name="cai-fix-suggestion-preventive-fix",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[FixSuggestion],
            system_prompt="""
            You are an expert software engineer who specializes in preventive measures
            and long-term solutions. Your task is to suggest improvements that prevent
            similar failures in the future.

            For each suggestion:
            1. Identify the underlying patterns that led to the failure
            2. Suggest structural improvements
            3. Recommend additional safeguards
            4. Propose testing enhancements
            5. Consider maintainability improvements

            Focus on:
            - Robust error handling and validation
            - Improved test coverage and quality
            - Code structure and architecture improvements
            - Documentation and clarity enhancements

            Consider:
            - Team workflow improvements
            - Tooling and automation opportunities
            - Code review and quality gates
            - Monitoring and observability
            """,
        )

        # Best practice agent
        self.best_practice_agent = make_agent_runner(
            worker_name="cai-fix-suggestion-best-practice",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[FixSuggestion],
            system_prompt="""
            You are an expert software engineer who specializes in industry best practices
            and code quality. Your task is to suggest improvements that align with
            modern software development standards.

            For each suggestion:
            1. Reference specific best practices or patterns
            2. Explain the long-term benefits
            3. Provide examples of proper implementation
            4. Consider team adoption and learning curve
            5. Align with project conventions

            Focus on:
            - SOLID principles and clean code practices
            - Design patterns and architectural improvements
            - Type safety and error handling
            - Performance and scalability considerations
            - Testing strategies and methodologies

            Consider:
            - Language-specific idioms and conventions
            - Framework-specific best practices
            - Industry standards and guidelines
            - Tool and library recommendations
            """,
        )

    def _initialize_solution_library(self) -> dict[str, Any]:
        """Initialize the solution library with common fixes."""
        return {
            "assertion_fixes": {
                "none_instead_of_value": [
                    {
                        "type": "code_change",
                        "description": "Add missing return statement",
                        "priority": "high",
                        "effort": "trivial",
                        "template": "return {expected_value}",
                    },
                    {
                        "type": "code_change",
                        "description": "Initialize variable with default value",
                        "priority": "medium",
                        "effort": "low",
                        "template": "{variable} = {default_value} if {variable} is None else {variable}",
                    },
                ],
                "type_mismatch": [
                    {
                        "type": "code_change",
                        "description": "Add explicit type conversion",
                        "priority": "high",
                        "effort": "trivial",
                        "template": "{target_type}({value})",
                    },
                    {
                        "type": "code_change",
                        "description": "Update type annotations",
                        "priority": "medium",
                        "effort": "low",
                        "template": "def function_name({param}: {correct_type}) -> {return_type}:",
                    },
                ],
            },
            "import_fixes": {
                "missing_dependency": [
                    {
                        "type": "dependency_update",
                        "description": "Install missing package",
                        "priority": "critical",
                        "effort": "trivial",
                        "template": "pip install {package_name}",
                    },
                    {
                        "type": "configuration_change",
                        "description": "Add to requirements.txt",
                        "priority": "high",
                        "effort": "trivial",
                        "template": "echo '{package_name}' >> requirements.txt",
                    },
                ],
                "circular_import": [
                    {
                        "type": "code_change",
                        "description": "Use deferred imports",
                        "priority": "high",
                        "effort": "medium",
                        "template": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from {module} import {class}",
                    },
                    {
                        "type": "code_change",
                        "description": "Reorganize module structure",
                        "priority": "medium",
                        "effort": "high",
                        "template": "Extract common dependencies to separate module",
                    },
                ],
            },
            "runtime_fixes": {
                "attribute_error": [
                    {
                        "type": "code_change",
                        "description": "Add attribute existence check",
                        "priority": "high",
                        "effort": "low",
                        "template": "if hasattr({object}, '{attribute}'):\n    {object}.{attribute}",
                    },
                    {
                        "type": "code_change",
                        "description": "Use getattr with default",
                        "priority": "medium",
                        "effort": "trivial",
                        "template": "getattr({object}, '{attribute}', {default_value})",
                    },
                ],
                "key_error": [
                    {
                        "type": "code_change",
                        "description": "Use dict.get() with default",
                        "priority": "high",
                        "effort": "trivial",
                        "template": "{dict}.get('{key}', {default_value})",
                    },
                    {
                        "type": "code_change",
                        "description": "Add key existence check",
                        "priority": "medium",
                        "effort": "low",
                        "template": "if '{key}' in {dict}:\n    {dict}['{key}']",
                    },
                ],
            },
        }

    async def generate_fix_suggestions(
        self,
        context: FixContext,
        max_suggestions: int = 10,
        minimum_confidence: float = 0.1,
        include_preventive: bool = True,
        include_best_practices: bool = True,
    ) -> list[FixSuggestion]:
        """
        Generate comprehensive fix suggestions for a test failure.

        Args:
            context: Fix context with failure data and analysis results
            max_suggestions: Maximum number of suggestions to generate
            minimum_confidence: Minimum confidence threshold
            include_preventive: Whether to include preventive suggestions
            include_best_practices: Whether to include best practice suggestions

        Returns:
            List of fix suggestions ranked by priority and confidence
        """
        start_time = datetime.now(UTC)

        try:
            self.logger.info(
                f"Generating fix suggestions for failure {context.failure_data.failure_id}"
            )

            # Check cache first
            cache_key = self._create_cache_key(context)
            if cache_key in self._suggestion_cache:
                self.logger.info("Using cached fix suggestions")
                return self._suggestion_cache[cache_key]

            all_suggestions: list[FixSuggestion] = []

            with llm_request_context(schema_id=SCHEMA_TEST_FAILURE):
                # 1. Generate immediate fixes
                immediate_fixes = await self._generate_immediate_fixes(context)
                all_suggestions.extend(immediate_fixes)

                # 2. Generate preventive fixes
                if include_preventive:
                    preventive_fixes = await self._generate_preventive_fixes(context)
                    all_suggestions.extend(preventive_fixes)

                # 3. Generate best practice suggestions
                if include_best_practices:
                    best_practice_fixes = await self._generate_best_practice_fixes(context)
                    all_suggestions.extend(best_practice_fixes)

            # 4. Generate pattern-based fixes (no LLM call)
            pattern_fixes = await self._generate_pattern_based_fixes(context)
            all_suggestions.extend(pattern_fixes)

            # Consolidate and rank suggestions
            consolidated_suggestions = self._consolidate_suggestions(all_suggestions)

            # Filter by confidence and limit results
            filtered_suggestions = [
                suggestion
                for suggestion in consolidated_suggestions
                if suggestion.confidence_score >= minimum_confidence
            ][:max_suggestions]

            # Cache results
            self._suggestion_cache[cache_key] = filtered_suggestions

            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            self.logger.info(
                "Generated %d fix suggestions in %.2fs (inference_id=%s, failover_chain_id=%s)",
                len(filtered_suggestions),
                processing_time,
                last_inference_id(),
                last_failover_chain_id(),
            )

            return filtered_suggestions

        except Exception as e:
            self.logger.error(f"Fix suggestion generation failed: {str(e)}")

            # Return fallback suggestions
            return self._create_fallback_suggestions(context)

    async def _generate_immediate_fixes(self, context: FixContext) -> list[FixSuggestion]:
        """Generate immediate fix suggestions that directly address root causes."""
        try:
            analysis_input = self._prepare_immediate_fix_input(context)

            result = await self.immediate_fix_agent.run(
                f"Generate immediate fixes for this test failure:\n\n{analysis_input}"
            )

            # Validate and enhance results
            enhanced_suggestions = []
            for suggestion in result.output:
                enhanced_suggestion = self._enhance_suggestion_with_context(
                    suggestion,
                    context,
                    "immediate",
                )
                enhanced_suggestions.append(enhanced_suggestion)

            return enhanced_suggestions

        except Exception as e:
            self.logger.error(f"Immediate fix generation failed: {str(e)}")
            return []

    async def _generate_preventive_fixes(self, context: FixContext) -> list[FixSuggestion]:
        """Generate preventive fix suggestions to avoid similar failures."""
        try:
            analysis_input = self._prepare_preventive_fix_input(context)

            result = await self.preventive_fix_agent.run(
                f"Generate preventive fixes for this test failure:\n\n{analysis_input}"
            )

            # Validate and enhance results
            enhanced_suggestions = []
            for suggestion in result.output:
                enhanced_suggestion = self._enhance_suggestion_with_context(
                    suggestion,
                    context,
                    "preventive",
                )
                enhanced_suggestions.append(enhanced_suggestion)

            return enhanced_suggestions

        except Exception as e:
            self.logger.error(f"Preventive fix generation failed: {str(e)}")
            return []

    async def _generate_best_practice_fixes(self, context: FixContext) -> list[FixSuggestion]:
        """Generate best practice fix suggestions for code quality improvements."""
        try:
            analysis_input = self._prepare_best_practice_input(context)

            result = await self.best_practice_agent.run(
                f"Generate best practice improvements for this test failure:\n\n{analysis_input}"
            )

            # Validate and enhance results
            enhanced_suggestions = []
            for suggestion in result.output:
                enhanced_suggestion = self._enhance_suggestion_with_context(
                    suggestion,
                    context,
                    "best_practice",
                )
                enhanced_suggestions.append(enhanced_suggestion)

            return enhanced_suggestions

        except Exception as e:
            self.logger.error(f"Best practice fix generation failed: {str(e)}")
            return []

    async def _generate_pattern_based_fixes(self, context: FixContext) -> list[FixSuggestion]:
        """Generate fix suggestions based on known patterns and solutions."""
        suggestions = []

        try:
            error_message = context.error_details.error_message.lower()

            # Look for pattern matches in solution library
            for category, patterns in self._solution_library.items():
                for pattern_name, solutions in patterns.items():
                    if self._matches_error_pattern(error_message, pattern_name):
                        for solution in solutions:
                            suggestion = self._create_pattern_based_suggestion(
                                pattern_name, solution, context
                            )
                            suggestions.append(suggestion)

            return suggestions

        except Exception as e:
            self.logger.error(f"Pattern-based fix generation failed: {str(e)}")
            return []

    def _prepare_immediate_fix_input(self, context: FixContext) -> str:
        """Prepare input for immediate fix generation."""
        parts = [
            f"Test Name: {context.failure_data.context.test_name}",
            f"Error Type: {context.error_details.error_type}",
            f"Error Message: {context.error_details.error_message}",
        ]

        if context.error_details.file_path:
            parts.append(f"File: {context.error_details.file_path}")

        if context.error_details.line_number:
            parts.append(f"Line: {context.error_details.line_number}")

        if context.root_causes:
            parts.append("Root Causes:")
            for cause in context.root_causes:
                parts.append(f"- {cause.description} (confidence: {cause.confidence_score:.2f})")

        if context.code_context:
            parts.append("Code Context:")
            for file_path, code in context.code_context.items():
                parts.append(f"{file_path}:\n{code}")

        return "\n".join(parts)

    def _prepare_preventive_fix_input(self, context: FixContext) -> str:
        """Prepare input for preventive fix generation."""
        parts = [
            f"Test Name: {context.failure_data.context.test_name}",
            f"Error Type: {context.error_details.error_type}",
            f"Framework: {context.failure_data.context.framework}",
        ]

        if context.root_causes:
            parts.append("Root Causes:")
            for cause in context.root_causes:
                parts.append(f"- {cause.category}: {cause.description}")

        if context.historical_fixes:
            parts.append("Historical Fixes:")
            for fix in context.historical_fixes:
                parts.append(f"- {fix.get('description', 'No description')}")

        if context.project_structure:
            parts.append(f"Project Structure: {json.dumps(context.project_structure, indent=2)}")

        return "\n".join(parts)

    def _prepare_best_practice_input(self, context: FixContext) -> str:
        """Prepare input for best practice fix generation."""
        parts = [
            f"Test Name: {context.failure_data.context.test_name}",
            f"Framework: {context.failure_data.context.framework}",
            f"Error Type: {context.error_details.error_type}",
        ]

        if context.code_context:
            parts.append("Code Context:")
            for file_path, code in context.code_context.items():
                parts.append(f"{file_path}:\n{code}")

        if context.team_preferences:
            parts.append(f"Team Preferences: {json.dumps(context.team_preferences, indent=2)}")

        return "\n".join(parts)

    def _enhance_suggestion_with_context(
        self, suggestion: FixSuggestion, context: FixContext, suggestion_type: str
    ) -> FixSuggestion:
        """Enhance a suggestion with additional context and validation."""
        # Ensure unique ID
        suggestion.suggestion_id = f"{suggestion_type}_{str(uuid.uuid4())[:8]}"

        # Adjust confidence based on suggestion type
        if suggestion_type == "immediate":
            # Immediate fixes are typically more reliable
            suggestion.confidence_score = min(suggestion.confidence_score * 1.1, 1.0)
        elif suggestion_type == "preventive":
            # Preventive fixes are valuable but less certain
            suggestion.confidence_score = suggestion.confidence_score * 0.8
        elif suggestion_type == "best_practice":
            # Best practice suggestions are good but not urgent
            suggestion.confidence_score = suggestion.confidence_score * 0.7

        # Map confidence score to level
        if suggestion.confidence_score >= 0.9:
            suggestion.confidence = ConfidenceLevel.VERY_HIGH
        elif suggestion.confidence_score >= 0.75:
            suggestion.confidence = ConfidenceLevel.HIGH
        elif suggestion.confidence_score >= 0.5:
            suggestion.confidence = ConfidenceLevel.MEDIUM
        elif suggestion.confidence_score >= 0.25:
            suggestion.confidence = ConfidenceLevel.LOW
        else:
            suggestion.confidence = ConfidenceLevel.VERY_LOW

        # Add context-specific information
        if context.error_details.file_path and not suggestion.files_to_modify:
            suggestion.files_to_modify = [context.error_details.file_path]

        # Add verification steps if missing
        if not suggestion.verification_steps:
            suggestion.verification_steps = [
                "Run the specific test that was failing",
                "Run the full test suite to ensure no regressions",
                "Verify the fix addresses the root cause",
            ]

        return suggestion

    def _matches_error_pattern(self, error_message: str, pattern_name: str) -> bool:
        """Check if error message matches a known pattern."""
        pattern_keywords = {
            "none_instead_of_value": ["none", "expected", "actual"],
            "type_mismatch": ["type", "expected", "actual", "class"],
            "missing_dependency": ["no module named", "import", "modulenotfound"],
            "circular_import": ["circular", "import", "partially initialized"],
            "attribute_error": ["attribute", "object", "has no"],
            "key_error": ["key", "keyerror"],
        }

        keywords = pattern_keywords.get(pattern_name, [])
        return any(keyword in error_message for keyword in keywords)

    def _create_pattern_based_suggestion(
        self, pattern_name: str, solution: dict[str, Any], context: FixContext
    ) -> FixSuggestion:
        """Create a fix suggestion based on a known pattern and solution."""
        suggestion = FixSuggestion(
            suggestion_id=f"pattern_{pattern_name}_{str(uuid.uuid4())[:8]}",
            title=solution.get("description", f"Fix for {pattern_name}"),
            description=solution.get("description", f"Apply standard fix for {pattern_name}"),
            fix_type=solution.get("type", "code_change"),
            priority=solution.get("priority", "medium"),
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.7,
            estimated_effort=solution.get("effort", "medium"),
        )

        # Add code example if template exists
        if "template" in solution:
            suggestion.code_examples = [solution["template"]]

        return suggestion

    def _consolidate_suggestions(self, suggestions: list[FixSuggestion]) -> list[FixSuggestion]:
        """Consolidate similar suggestions and rank by priority and confidence."""
        if not suggestions:
            return []

        # Group similar suggestions
        consolidated = {}
        for suggestion in suggestions:
            # Use title and fix_type as grouping key
            key = f"{suggestion.fix_type}_{suggestion.title[:30]}"

            if key not in consolidated:
                consolidated[key] = suggestion
            else:
                # Merge suggestions
                existing = consolidated[key]

                # Combine descriptions (respect max_length=2000 from model)
                if suggestion.description not in existing.description:
                    combined = (
                        f"{existing.description}\n\nAlternative approach: {suggestion.description}"
                    )
                    existing.description = combined[:2000]

                # Combine code examples
                if suggestion.code_examples:
                    if not existing.code_examples:
                        existing.code_examples = []
                    existing.code_examples.extend(suggestion.code_examples)

                # Use higher confidence
                if suggestion.confidence_score > existing.confidence_score:
                    existing.confidence_score = suggestion.confidence_score
                    existing.confidence = suggestion.confidence

                # Use higher priority
                priority_order = ["critical", "high", "medium", "low"]
                if priority_order.index(suggestion.priority) < priority_order.index(
                    existing.priority
                ):
                    existing.priority = suggestion.priority

        # Sort by priority and confidence
        priority_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        sorted_suggestions = sorted(
            consolidated.values(),
            key=lambda x: (priority_weights.get(x.priority, 0), x.confidence_score),
            reverse=True,
        )

        return sorted_suggestions

    def _create_cache_key(self, context: FixContext) -> str:
        """Create a cache key for the fix context."""
        key_parts = [
            context.failure_data.failure_id,
            context.error_details.error_type
            if isinstance(context.error_details.error_type, str)
            else context.error_details.error_type.value,
            context.error_details.error_message[:100],
            str(len(context.root_causes)),
        ]

        return "|".join(key_parts)

    def _create_fallback_suggestions(self, context: FixContext) -> list[FixSuggestion]:
        """Create fallback suggestions when AI generation fails."""
        fallback_suggestion = FixSuggestion(
            suggestion_id=f"fallback_{str(uuid.uuid4())[:8]}",
            title="Review and debug the test failure",
            description="Carefully examine the error message, traceback, and test code to identify the specific issue causing the failure.",
            fix_type="investigation_needed",
            priority="medium",
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.3,
            estimated_effort="medium",
            verification_steps=[
                "Analyze the error message and traceback",
                "Review the test code and implementation",
                "Check for recent changes that might have caused the issue",
                "Run the test in isolation to reproduce the failure",
            ],
        )

        return [fallback_suggestion]
