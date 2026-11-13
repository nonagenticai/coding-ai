"""
PydanticAI-powered intelligent test failure analyzer.

This module provides AI-powered analysis of test failures using PydanticAI for:
- Intelligent root cause identification
- Pattern detection and classification
- Contextual fix suggestions
- Confidence scoring and reasoning
- Multi-language test failure support

The analyzer uses advanced AI models to understand test failures beyond simple
pattern matching, providing insights into complex failure scenarios and suggesting
actionable solutions.
"""

import logging
import re
import uuid
from datetime import UTC, datetime
from typing import Literal

from coding_client.llm import (
    last_failover_chain_id,
    last_inference_id,
    llm_request_context,
)

from models.ai_test_failure_models import (
    AIAnalysisResult,
    AssertionDetails,
    ConfidenceLevel,
    ErrorDetails,
    FailureAnalysisRequest,
    FailureAnalysisResponse,
    FailureType,
    FixSuggestion,
    RootCause,
    TestFailureData,
    TestFailurePattern,
)
from services.llm_agents import SCHEMA_TEST_FAILURE, make_agent_runner

logger = logging.getLogger(__name__)


class LLMAgentUnavailableError(Exception):
    """Raised when an LLM-backed analysis agent cannot produce a result.

    Signals that the LLM plane was unreachable or the agent errored — as opposed
    to an agent that ran successfully and simply found nothing. The former is a
    degraded/outage condition; the latter is a genuine empty result. Callers use
    this to set ``degraded=True`` on the response instead of silently emitting
    fabricated fallback content as a clean success.
    """


# Type alias for assertion types matching the AssertionDetails model
AssertionTypeLiteral = Literal[
    "assertEqual",
    "assertNotEqual",
    "assertTrue",
    "assertFalse",
    "assertIs",
    "assertIsNot",
    "assertIsNone",
    "assertIsNotNone",
    "assertIn",
    "assertNotIn",
    "assertRaises",
    "assertRaisesRegex",
    "assertAlmostEqual",
    "assertNotAlmostEqual",
    "assertGreater",
    "assertGreaterEqual",
    "assertLess",
    "assertLessEqual",
    "assertRegex",
    "assertNotRegex",
    "assertCountEqual",
    "assertMultiLineEqual",
    "assertSequenceEqual",
    "assertListEqual",
    "assertTupleEqual",
    "assertSetEqual",
    "assertDictEqual",
    "custom",
    "unknown",
]


class AITestFailureAnalyzer:
    """
    AI-powered test failure analyzer using PydanticAI.

    This analyzer provides intelligent analysis of test failures by:
    1. Parsing and understanding test output
    2. Identifying root causes using AI reasoning
    3. Detecting patterns and classifying failures
    4. Generating contextual fix suggestions
    5. Providing confidence scores and explanations
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the AI test failure analyzer.

        Args:
            model_name: Retained for backwards compatibility; ignored. Provider
                selection is now server-side in inference-router.
        """
        # ``model_name`` is kept on the instance for log/diagnostic continuity
        # only — the actual schema_id used at the LLM plane is the registry
        # entry ``test_failure_analysis``.
        self.model_name = model_name or SCHEMA_TEST_FAILURE

        self.logger = logging.getLogger(__name__)
        self.is_available = True

        # Initialize AI agents for different analysis tasks
        try:
            self._init_ai_agents()
        except Exception as e:
            self.logger.warning(f"Failed to initialize AI agents: {e}")
            self.is_available = False

        # Pattern cache for common failure patterns
        self._pattern_cache: dict[str, TestFailurePattern] = {}

        # Known failure signatures for quick classification
        self._known_signatures: set[str] = set()

    def _init_ai_agents(self) -> None:
        """Initialize PydanticAI agents for different analysis tasks."""

        # Root cause analysis agent
        self.root_cause_agent = make_agent_runner(
            worker_name="cai-test-failure-analyzer-root-cause",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[RootCause],
            system_prompt="""
            You are an expert software engineer specializing in test failure analysis.
            Your task is to analyze test failures and identify the most likely root causes.

            For each root cause you identify:
            1. Provide a clear, actionable description
            2. Assign appropriate severity and confidence levels
            3. Include specific evidence from the test output
            4. Consider the broader context of the failure

            Focus on:
            - Logic errors and implementation bugs
            - Configuration and environment issues
            - Dependency and integration problems
            - Data validation and type safety issues
            - Race conditions and timing issues

            Be thorough but concise. Rank causes by likelihood and impact.
            """,
        )

        # Fix suggestion agent
        self.fix_suggestion_agent = make_agent_runner(
            worker_name="cai-test-failure-analyzer-fix-suggestion",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[FixSuggestion],
            system_prompt="""
            You are an expert software engineer who excels at diagnosing and resolving test failures.
            Your task is to generate actionable fix suggestions based on test failure analysis.

            For each suggestion:
            1. Provide a clear, specific action to take
            2. Include code examples when appropriate
            3. Explain why this fix addresses the root cause
            4. Estimate the effort and potential side effects
            5. Suggest verification steps

            Focus on:
            - Immediate fixes for the specific failure
            - Preventive measures to avoid similar issues
            - Best practices and code quality improvements
            - Testing and validation improvements

            Prioritize suggestions by impact and feasibility.
            """,
        )

        # Pattern detection agent
        self.pattern_agent = make_agent_runner(
            worker_name="cai-test-failure-analyzer-pattern",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[TestFailurePattern],
            system_prompt="""
            You are an expert in test failure pattern recognition and classification.
            Your task is to identify recurring patterns in test failures.

            For each pattern:
            1. Create a unique signature that captures the essence
            2. Classify the pattern type accurately
            3. Provide confidence in the pattern detection
            4. Include typical causes and common fixes

            Look for patterns in:
            - Error messages and stack traces
            - Assertion failures and data mismatches
            - Timing and execution flow issues
            - Environment and configuration problems
            - Dependency and version conflicts

            Consider both syntactic patterns (text matching) and semantic patterns (meaning).
            """,
        )

    async def analyze_failure(self, request: FailureAnalysisRequest) -> FailureAnalysisResponse:
        """
        Perform comprehensive AI-powered analysis of a test failure.

        Args:
            request: Test failure analysis request

        Returns:
            Complete analysis response with results or error information
        """
        start_time = datetime.now(UTC)

        try:
            self.logger.info(f"Starting AI analysis for request {request.request_id}")

            # Attach the registry schema_id to all LLM-plane calls made within
            # this analysis. Tenant/job/license headers are inherited from
            # whatever outer context the caller (FastAPI router) installed.
            with llm_request_context(schema_id=SCHEMA_TEST_FAILURE):
                return await self._analyze_failure_inner(request, start_time)

        except Exception as e:
            self.logger.error(f"AI analysis failed for request {request.request_id}: {str(e)}")

            return FailureAnalysisResponse(
                request_id=request.request_id,
                success=False,
                error_message=f"Analysis failed: {str(e)}",
                processing_time=(datetime.now(UTC) - start_time).total_seconds(),
            )

    async def _analyze_failure_inner(
        self,
        request: FailureAnalysisRequest,
        start_time: datetime,
    ) -> FailureAnalysisResponse:
        """Inner body of :meth:`analyze_failure`, executed under the request context."""
        try:
            # Extract and parse error details
            error_details = await self._extract_error_details(request.failure_data)

            # Extract assertion details if applicable
            assertion_details = await self._extract_assertion_details(request.failure_data)

            # Each LLM-backed stage is attempted independently. A stage whose
            # agent is unreachable/errors raises LLMAgentUnavailableError; we
            # record that as `degraded` and leave that stage's results EMPTY
            # rather than fabricating fallback content and reporting a clean
            # success. A stage that runs and genuinely finds nothing leaves
            # `degraded` False — that is a real (non-degraded) empty result.
            degraded = False

            # Perform AI-powered root cause analysis
            root_causes = []
            if request.max_root_causes > 0:
                try:
                    root_causes = await self._analyze_root_causes(
                        request.failure_data,
                        error_details,
                        request.max_root_causes,
                        request.minimum_confidence,
                    )
                except LLMAgentUnavailableError as agent_error:
                    degraded = True
                    self.logger.warning(
                        "Root-cause agent unavailable for request %s (LLM outage): %s",
                        request.request_id,
                        str(agent_error),
                    )

            # Generate fix suggestions
            fix_suggestions = []
            if request.include_suggestions and request.max_suggestions > 0:
                try:
                    fix_suggestions = await self._generate_fix_suggestions(
                        request.failure_data,
                        error_details,
                        root_causes,
                        request.max_suggestions,
                        request.minimum_confidence,
                    )
                except LLMAgentUnavailableError as agent_error:
                    degraded = True
                    self.logger.warning(
                        "Fix-suggestion agent unavailable for request %s (LLM outage): %s",
                        request.request_id,
                        str(agent_error),
                    )

            # Detect patterns
            patterns = []
            if request.include_patterns:
                try:
                    patterns = await self._detect_patterns(
                        request.failure_data, error_details, request.minimum_confidence
                    )
                except LLMAgentUnavailableError as agent_error:
                    degraded = True
                    self.logger.warning(
                        "Pattern agent unavailable for request %s (LLM outage): %s",
                        request.request_id,
                        str(agent_error),
                    )

            # Calculate overall analysis confidence
            analysis_confidence, analysis_confidence_score = self._calculate_overall_confidence(
                root_causes, fix_suggestions, patterns
            )

            # Create analysis result. Carry the degraded flag in metadata so it
            # survives the service layer (which returns analysis_result, not this
            # response) all the way to the router.
            analysis_result = AIAnalysisResult(
                analysis_id=str(uuid.uuid4()),
                failure_data=request.failure_data,
                error_details=error_details,
                assertion_details=assertion_details,
                root_causes=root_causes,
                fix_suggestions=fix_suggestions,
                detected_patterns=patterns,
                analysis_confidence=analysis_confidence,
                analysis_confidence_score=analysis_confidence_score,
                processing_time=(datetime.now(UTC) - start_time).total_seconds(),
                ai_model_version=self.model_name,
                additional_metadata={"degraded": True} if degraded else None,
            )

            self.logger.info(
                "AI analysis completed for request %s with %d root causes "
                "and %d suggestions (degraded=%s, inference_id=%s, failover_chain_id=%s)",
                request.request_id,
                len(root_causes),
                len(fix_suggestions),
                degraded,
                last_inference_id(),
                last_failover_chain_id(),
            )

            return FailureAnalysisResponse(
                request_id=request.request_id,
                success=True,
                degraded=degraded,
                analysis_result=analysis_result,
                processing_time=(datetime.now(UTC) - start_time).total_seconds(),
            )

        except Exception as e:
            self.logger.error(f"AI analysis failed for request {request.request_id}: {str(e)}")

            return FailureAnalysisResponse(
                request_id=request.request_id,
                success=False,
                error_message=f"Analysis failed: {str(e)}",
                processing_time=(datetime.now(UTC) - start_time).total_seconds(),
            )

    async def _extract_error_details(self, failure_data: TestFailureData) -> ErrorDetails:
        """Extract detailed error information from test failure data."""
        raw_output = failure_data.raw_output

        # Determine error type
        error_type = self._classify_error_type(raw_output)

        # Extract error message
        error_message = self._extract_error_message(raw_output)

        # Extract traceback
        traceback_text = self._extract_traceback(raw_output)

        # Extract file and line information
        file_path, line_number, column_number = self._extract_location_info(raw_output)

        # Parse stack trace
        stack_trace = self._parse_stack_trace(traceback_text) if traceback_text else None

        return ErrorDetails(
            error_type=error_type,
            error_message=error_message,
            traceback=traceback_text,
            file_path=file_path,
            line_number=line_number,
            column_number=column_number,
            stack_trace=stack_trace,
        )

    async def _extract_assertion_details(
        self, failure_data: TestFailureData
    ) -> AssertionDetails | None:
        """Extract assertion-specific details if the failure is an assertion error."""
        raw_output = failure_data.raw_output

        if (
            "AssertionError" not in raw_output
            and "AssertionError" not in raw_output
            and "assert" not in raw_output.lower()
        ):
            return None

        # Extract assertion type
        assertion_type = self._classify_assertion_type(raw_output)

        # Extract expected and actual values
        expected_value, actual_value = self._extract_assertion_values(raw_output)

        # Extract comparison operator
        comparison_operator = self._extract_comparison_operator(raw_output)

        # Extract assertion message
        assertion_message = self._extract_assertion_message(raw_output)

        # Calculate value difference
        value_difference = None
        if expected_value and actual_value:
            value_difference = self._calculate_value_difference(expected_value, actual_value)

        return AssertionDetails(
            assertion_type=assertion_type,
            expected_value=expected_value,
            actual_value=actual_value,
            comparison_operator=comparison_operator,
            assertion_message=assertion_message,
            value_difference=value_difference,
        )

    async def _analyze_root_causes(
        self,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
        max_causes: int,
        minimum_confidence: float,
    ) -> list[RootCause]:
        """Use AI to analyze and identify root causes of the test failure."""
        try:
            # Prepare context for AI analysis
            context = self._prepare_analysis_context(failure_data, error_details)

            # Run AI analysis
            result = await self.root_cause_agent.run(
                f"Analyze this test failure and identify the root causes (max: {max_causes}, min confidence: {minimum_confidence}):\n\n{context}",
            )

            # Filter and validate results
            valid_causes: list[RootCause] = []
            for cause in result.output:
                if cause.confidence_score >= minimum_confidence:
                    # Ensure unique IDs
                    cause.cause_id = f"cause_{len(valid_causes) + 1}_{str(uuid.uuid4())[:8]}"
                    valid_causes.append(cause)

                    if len(valid_causes) >= max_causes:
                        break

            return valid_causes

        except Exception as e:
            self.logger.error(f"Root cause analysis failed: {str(e)}")
            # The LLM agent was unreachable/errored. Signal a degraded
            # condition rather than fabricating a fallback cause and
            # reporting it as a clean success.
            raise LLMAgentUnavailableError(f"root-cause agent failed: {e}") from e

    async def _generate_fix_suggestions(
        self,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
        root_causes: list[RootCause],
        max_suggestions: int,
        minimum_confidence: float,
    ) -> list[FixSuggestion]:
        """Use AI to generate fix suggestions for the test failure."""
        try:
            # Prepare context for AI analysis
            context = self._prepare_fix_context(failure_data, error_details, root_causes)

            # Run AI analysis
            result = await self.fix_suggestion_agent.run(
                f"Generate fix suggestions for this test failure (max: {max_suggestions}, min confidence: {minimum_confidence}):\n\n{context}",
            )

            # Filter and validate results
            valid_suggestions: list[FixSuggestion] = []
            for suggestion in result.output:
                if suggestion.confidence_score >= minimum_confidence:
                    # Ensure unique IDs
                    suggestion.suggestion_id = (
                        f"fix_{len(valid_suggestions) + 1}_{str(uuid.uuid4())[:8]}"
                    )
                    valid_suggestions.append(suggestion)

                    if len(valid_suggestions) >= max_suggestions:
                        break

            return valid_suggestions

        except Exception as e:
            self.logger.error(f"Fix suggestion generation failed: {str(e)}")
            raise LLMAgentUnavailableError(f"fix-suggestion agent failed: {e}") from e

    async def _detect_patterns(
        self,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
        minimum_confidence: float,
    ) -> list[TestFailurePattern]:
        """Use AI to detect patterns in the test failure."""
        try:
            # Check pattern cache first
            signature = self._create_failure_signature(failure_data, error_details)
            if signature in self._pattern_cache:
                cached_pattern = self._pattern_cache[signature]
                cached_pattern.frequency += 1
                return [cached_pattern]

            # Prepare context for AI analysis
            context = self._prepare_pattern_context(failure_data, error_details)

            # Run AI analysis
            result = await self.pattern_agent.run(
                f"Detect patterns in this test failure (min confidence: {minimum_confidence}):\n\n{context}",
            )

            # Filter and validate results
            valid_patterns: list[TestFailurePattern] = []
            for pattern in result.output:
                if pattern.confidence_score >= minimum_confidence:
                    # Ensure unique IDs
                    pattern.pattern_id = (
                        f"pattern_{len(valid_patterns) + 1}_{str(uuid.uuid4())[:8]}"
                    )
                    valid_patterns.append(pattern)

                    # Cache the pattern
                    self._pattern_cache[signature] = pattern
                    self._known_signatures.add(signature)

            return valid_patterns

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}")
            raise LLMAgentUnavailableError(f"pattern agent failed: {e}") from e

    def _classify_error_type(self, output: str) -> FailureType:
        """Classify the type of error based on the output."""
        output_lower = output.lower()

        if "assertionerror" in output_lower or "assertion" in output_lower:
            return FailureType.ASSERTION_ERROR
        elif "syntaxerror" in output_lower or "syntax" in output_lower:
            return FailureType.SYNTAX_ERROR
        elif "importerror" in output_lower or "modulenotfounderror" in output_lower:
            return FailureType.IMPORT_ERROR
        elif "timeout" in output_lower or "timed out" in output_lower:
            return FailureType.TIMEOUT_ERROR
        elif "memoryerror" in output_lower or "out of memory" in output_lower:
            return FailureType.MEMORY_ERROR
        elif "connectionerror" in output_lower or "network" in output_lower:
            return FailureType.NETWORK_ERROR
        elif "dependency" in output_lower or "package" in output_lower:
            return FailureType.DEPENDENCY_ERROR
        elif "configuration" in output_lower or "config" in output_lower:
            return FailureType.CONFIGURATION_ERROR
        elif any(error in output_lower for error in ["error", "exception", "failed"]):
            return FailureType.RUNTIME_ERROR
        else:
            return FailureType.UNKNOWN_ERROR

    def _extract_error_message(self, output: str) -> str:
        """Extract the primary error message from the output."""
        lines = output.split("\n")

        # Look for common error patterns
        error_patterns = [
            r"(\w+Error): (.+?)(?:[\r\n]|$)",
            r"AssertionError: (.+?)(?:[\r\n]|$)",
            r"FAILED.*?- (.+?)(?:[\r\n]|$)",
            r"ERROR.*?- (.+?)(?:[\r\n]|$)",
        ]

        for pattern in error_patterns:
            match = re.search(pattern, output, re.DOTALL)
            if match:
                return (
                    match.group(1).strip() if len(match.groups()) == 1 else match.group(2).strip()
                )

        # Fallback: look for lines that seem like error messages
        for line in lines:
            line = line.strip()
            if line and any(
                keyword in line.lower() for keyword in ["error", "failed", "exception"]
            ):
                return line

        return "Unknown error message"

    def _extract_traceback(self, output: str) -> str | None:
        """Extract traceback information from the output."""
        traceback_pattern = r"Traceback \(most recent call last\):(.*?)(?=\n\w+Error:|$)"
        match = re.search(traceback_pattern, output, re.DOTALL)

        if match:
            return match.group(1).strip()

        return None

    def _extract_location_info(self, output: str) -> tuple[str | None, int | None, int | None]:
        """Extract file path, line number, and column number from the output."""
        # Common patterns for location information
        location_patterns = [
            r'File "([^"]+)", line (\d+)(?:, in .+)?',
            r'File "([^"]+)", line (\d+), column (\d+)',
            r"([^:]+):(\d+):(\d+):",
            r"([^:]+):(\d+):",
        ]

        for pattern in location_patterns:
            match = re.search(pattern, output)
            if match:
                groups = match.groups()
                file_path = groups[0] if groups[0] else None
                line_number = int(groups[1]) if len(groups) > 1 and groups[1] else None
                column_number = int(groups[2]) if len(groups) > 2 and groups[2] else None
                return file_path, line_number, column_number

        return None, None, None

    def _parse_stack_trace(self, traceback_text: str) -> list[str]:
        """Parse stack trace into individual entries."""
        if not traceback_text:
            return []

        lines = traceback_text.split("\n")
        stack_entries = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("Traceback"):
                stack_entries.append(line)

        return stack_entries

    def _classify_assertion_type(self, output: str) -> AssertionTypeLiteral:
        """Classify the type of assertion that failed."""
        assertion_patterns: dict[AssertionTypeLiteral, str] = {
            "assertEqual": r"assertEqual",
            "assertNotEqual": r"assertNotEqual",
            "assertTrue": r"assertTrue",
            "assertFalse": r"assertFalse",
            "assertIs": r"assertIs",
            "assertIsNot": r"assertIsNot",
            "assertIsNone": r"assertIsNone",
            "assertIsNotNone": r"assertIsNotNone",
            "assertIn": r"assertIn",
            "assertNotIn": r"assertNotIn",
            "assertRaises": r"assertRaises",
            "assertAlmostEqual": r"assertAlmostEqual",
            "assertGreater": r"assertGreater",
            "assertLess": r"assertLess",
        }

        for assertion_type, pattern in assertion_patterns.items():
            if re.search(pattern, output):
                return assertion_type

        return "unknown"

    def _extract_assertion_values(self, output: str) -> tuple[str | None, str | None]:
        """Extract expected and actual values from assertion failure output."""
        # Common patterns for assertion values
        value_patterns = [
            r"Expected: (.+?)[\r\n]+Actual: (.+?)(?:[\r\n]|$)",
            r"expected: (.+?)[\r\n]+actual: (.+?)(?:[\r\n]|$)",
            r"!=[\s]*(.+?)[\r\n].*?==[\s]*(.+?)(?:[\r\n]|$)",
            r"AssertionError: (.+?) != (.+?)(?:[\r\n]|$)",
        ]

        for pattern in value_patterns:
            match = re.search(pattern, output, re.DOTALL)
            if match:
                return match.group(1).strip(), match.group(2).strip()

        return None, None

    def _extract_comparison_operator(self, output: str) -> str | None:
        """Extract the comparison operator from assertion failure output."""
        operators = ["==", "!=", "<", ">", "<=", ">=", "is", "is not", "in", "not in"]

        for op in operators:
            if op in output:
                return op

        return None

    def _extract_assertion_message(self, output: str) -> str | None:
        """Extract custom assertion message if present."""
        # Look for custom assertion messages
        message_patterns = [
            r"AssertionError: (.+?)(?:[\r\n]|$)",
            r"assert .+?, ['\"](.+?)['\"]",
        ]

        for pattern in message_patterns:
            match = re.search(pattern, output)
            if match:
                message = match.group(1).strip()
                # Filter out generic messages that are just value comparisons
                if not any(op in message for op in ["!=", "==", "<", ">", "<=", ">="]):
                    return message

        return None

    def _calculate_value_difference(self, expected: str, actual: str) -> str | None:
        """Calculate the difference between expected and actual values."""
        try:
            # Try to parse as numbers
            try:
                exp_num = float(expected)
                act_num = float(actual)
                num_diff = act_num - exp_num
                return f"Numerical difference: {num_diff:+.4f}"
            except ValueError:
                pass

            # Try character-by-character comparison for strings
            if len(expected) < 1000 and len(actual) < 1000:
                import difflib

                str_diff: list[str] = list(
                    difflib.unified_diff(
                        expected.splitlines(keepends=True),
                        actual.splitlines(keepends=True),
                        fromfile="expected",
                        tofile="actual",
                        lineterm="",
                    )
                )
                if str_diff:
                    return "".join(str_diff)

            # Simple length comparison
            if len(expected) != len(actual):
                return f"Length difference: expected {len(expected)}, actual {len(actual)}"

        except Exception:
            pass

        return None

    def _prepare_analysis_context(
        self, failure_data: TestFailureData, error_details: ErrorDetails
    ) -> str:
        """Prepare context information for AI analysis."""
        context_parts = [
            f"Test Failure ID: {failure_data.failure_id}",
            f"Test Name: {failure_data.context.test_name}",
            f"Framework: {failure_data.context.framework}",
            f"Error Type: {error_details.error_type}",
            f"Error Message: {error_details.error_message}",
        ]

        if error_details.file_path:
            context_parts.append(f"File: {error_details.file_path}")

        if error_details.line_number:
            context_parts.append(f"Line: {error_details.line_number}")

        if error_details.traceback:
            context_parts.append(f"Traceback:\n{error_details.traceback}")

        context_parts.append(f"Raw Output:\n{failure_data.raw_output}")

        return "\n".join(context_parts)

    def _prepare_fix_context(
        self,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
        root_causes: list[RootCause],
    ) -> str:
        """Prepare context information for fix suggestion generation."""
        context_parts = [
            f"Test Failure ID: {failure_data.failure_id}",
            f"Test Name: {failure_data.context.test_name}",
            f"Error Type: {error_details.error_type}",
            f"Error Message: {error_details.error_message}",
        ]

        if root_causes:
            context_parts.append("Identified Root Causes:")
            for cause in root_causes:
                context_parts.append(
                    f"- {cause.description} (confidence: {cause.confidence_score:.2f})"
                )

        context_parts.append(f"Raw Output:\n{failure_data.raw_output}")

        return "\n".join(context_parts)

    def _prepare_pattern_context(
        self, failure_data: TestFailureData, error_details: ErrorDetails
    ) -> str:
        """Prepare context information for pattern detection."""
        context_parts = [
            f"Error Type: {error_details.error_type}",
            f"Error Message: {error_details.error_message}",
            f"Framework: {failure_data.context.framework}",
        ]

        if error_details.stack_trace:
            context_parts.append("Stack Trace:")
            context_parts.extend(error_details.stack_trace)

        context_parts.append(f"Raw Output:\n{failure_data.raw_output}")

        return "\n".join(context_parts)

    def _create_failure_signature(
        self, failure_data: TestFailureData, error_details: ErrorDetails
    ) -> str:
        """Create a unique signature for the failure for pattern matching."""

        def _v(x):
            return x.value if hasattr(x, "value") else x

        signature_parts = [
            _v(error_details.error_type),
            error_details.error_message[:100],  # First 100 chars
            _v(failure_data.context.framework),
        ]

        if error_details.file_path:
            signature_parts.append(error_details.file_path.split("/")[-1])  # Just filename

        return "|".join(signature_parts)

    def _calculate_overall_confidence(
        self,
        root_causes: list[RootCause],
        fix_suggestions: list[FixSuggestion],
        patterns: list[TestFailurePattern],
    ) -> tuple[ConfidenceLevel, float]:
        """Calculate overall confidence in the analysis."""
        if not root_causes and not fix_suggestions and not patterns:
            return ConfidenceLevel.VERY_LOW, 0.0

        # Calculate weighted average confidence
        total_weight = 0.0
        weighted_sum = 0.0

        # Root causes have highest weight
        for cause in root_causes:
            weighted_sum += cause.confidence_score * 0.5
            total_weight += 0.5

        # Fix suggestions have medium weight
        for suggestion in fix_suggestions:
            weighted_sum += suggestion.confidence_score * 0.3
            total_weight += 0.3

        # Patterns have lower weight
        for pattern in patterns:
            weighted_sum += pattern.confidence_score * 0.2
            total_weight += 0.2

        if total_weight == 0:
            return ConfidenceLevel.VERY_LOW, 0.0

        avg_confidence = weighted_sum / total_weight

        # Map to confidence level
        if avg_confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH, avg_confidence
        elif avg_confidence >= 0.75:
            return ConfidenceLevel.HIGH, avg_confidence
        elif avg_confidence >= 0.5:
            return ConfidenceLevel.MEDIUM, avg_confidence
        elif avg_confidence >= 0.25:
            return ConfidenceLevel.LOW, avg_confidence
        else:
            return ConfidenceLevel.VERY_LOW, avg_confidence

    # NOTE: the former ``_fallback_root_cause_analysis`` /
    # ``_fallback_fix_suggestions`` / ``_fallback_pattern_detection`` helpers
    # were removed. They fabricated confidence-0.3 causes/suggestions/patterns
    # on an LLM outage and returned them as a clean success, hiding the outage.
    # Outage is now surfaced honestly: the LLM stages raise
    # LLMAgentUnavailableError, ``_analyze_failure_inner`` records
    # ``degraded=True`` and leaves those stages empty.
