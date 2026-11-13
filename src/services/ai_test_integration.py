"""
AI Test Integration Service for PydanticAI-powered test failure analysis.

This service provides seamless integration between the existing test execution
infrastructure and the new AI-powered test failure analysis system. It acts as
a bridge, translating between different data formats and orchestrating the
analysis workflow.

Features:
- Integration with existing TestExecutionService
- Data format translation and adaptation
- Unified analysis workflow
- Backward compatibility with existing systems
- Performance optimization and caching
- Error handling and graceful degradation
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from models.ai_test_failure_models import (
    AIAnalysisResult,
    FailureAnalysisRequest,
    TestFailureContext,
    TestFailureData,
    TestFileMetadata,
    TestFramework,
)
from models.testing_models import TestRequest, TestResult
from services.ai_failure_analysis import AIFailureAnalysisService

# TODO: TestExecutionService belongs to coding-testing, stubbed out
# from services.testing.test_execution_service import TestExecutionService

logger = logging.getLogger(__name__)


class _StubTestExecutionService:
    """Stub for TestExecutionService that belongs to coding-testing."""

    async def execute_test(self, request: TestRequest) -> TestResult:
        raise NotImplementedError(
            "TestExecutionService is not available in coding-ai. Use coding-testing service."
        )

    async def health_check(self) -> dict[str, Any]:
        return {"status": "stub", "message": "TestExecutionService not available in coding-ai"}


# Type alias for flexibility
TestExecutionService = _StubTestExecutionService


class AITestIntegrationService:
    """
    AI Test Integration Service.

    This service provides seamless integration between the existing test execution
    infrastructure and the new AI-powered test failure analysis system. It handles
    data translation, workflow orchestration, and maintains backward compatibility.
    """

    def __init__(
        self,
        test_execution_service: Any | None = None,
        ai_analysis_service: AIFailureAnalysisService | None = None,
        enable_ai_analysis: bool = True,
        analysis_timeout: float = 60.0,
        max_concurrent_analyses: int = 3,
    ) -> None:
        """
        Initialize the AI Test Integration Service.

        Args:
            test_execution_service: Existing test execution service (optional, stub used if None)
            ai_analysis_service: AI analysis service (optional, will create if not provided)
            enable_ai_analysis: Whether to enable AI analysis
            analysis_timeout: Timeout for AI analysis operations
            max_concurrent_analyses: Maximum concurrent AI analyses
        """
        self.test_execution_service = test_execution_service or _StubTestExecutionService()
        self.enable_ai_analysis = enable_ai_analysis
        self.analysis_timeout = analysis_timeout
        self.max_concurrent_analyses = max_concurrent_analyses

        # Initialize AI analysis service
        self.ai_analysis_service: AIFailureAnalysisService | None = None
        if ai_analysis_service:
            self.ai_analysis_service = ai_analysis_service
        elif enable_ai_analysis:
            self.ai_analysis_service = AIFailureAnalysisService(
                max_concurrent_analyses=max_concurrent_analyses,
                analysis_timeout=analysis_timeout,
            )

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Analysis semaphore for rate limiting
        self._analysis_semaphore = asyncio.Semaphore(max_concurrent_analyses)

        # Integration metrics
        self._metrics = {
            "total_tests": 0,
            "failed_tests": 0,
            "analyzed_failures": 0,
            "analysis_successes": 0,
            "analysis_failures": 0,
            "integration_errors": 0,
        }

    async def execute_test_with_analysis(
        self,
        request: TestRequest,
        enable_failure_analysis: bool = True,
        include_code_context: bool = True,
        include_suggestions: bool = True,
        include_patterns: bool = True,
    ) -> TestResult:
        """
        Execute a test and perform AI analysis on failures.

        Args:
            request: Test execution request
            enable_failure_analysis: Whether to analyze failures with AI
            include_code_context: Whether to include code context in analysis
            include_suggestions: Whether to include fix suggestions
            include_patterns: Whether to include pattern detection

        Returns:
            Test result with AI analysis if failure occurred
        """
        start_time = datetime.now(UTC)

        try:
            self.logger.info("Executing test with AI analysis capability")

            # Update metrics
            self._metrics["total_tests"] += 1

            # Execute the test using the existing service
            test_result = await self.test_execution_service.execute_test(request)

            # If test failed and AI analysis is enabled, perform analysis
            if (
                not test_result.success
                and enable_failure_analysis
                and self.enable_ai_analysis
                and self.ai_analysis_service
            ):
                self._metrics["failed_tests"] += 1

                try:
                    # Perform AI analysis
                    analysis_result = await self._analyze_test_failure(
                        test_result=test_result,
                        request=request,
                        include_code_context=include_code_context,
                        include_suggestions=include_suggestions,
                        include_patterns=include_patterns,
                    )

                    # Integrate analysis results into test result
                    if analysis_result:
                        test_result = self._integrate_analysis_results(test_result, analysis_result)
                        self._metrics["analysis_successes"] += 1
                    else:
                        self._metrics["analysis_failures"] += 1

                    self._metrics["analyzed_failures"] += 1

                except Exception as e:
                    self.logger.error(f"AI analysis failed: {str(e)}")
                    self._metrics["analysis_failures"] += 1

                    # Add error information to test result
                    test_result.test_metadata = getattr(test_result, "test_metadata", None) or {}
                    test_result.test_metadata["ai_analysis_error"] = str(e)

            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            self.logger.info(f"Test execution and analysis completed in {processing_time:.2f}s")

            return test_result

        except Exception as e:
            self.logger.error(f"Test execution with analysis failed: {str(e)}")
            self._metrics["integration_errors"] += 1
            raise

    async def analyze_existing_failure(
        self,
        test_result: TestResult,
        source_code: str | None = None,
        test_code: str | None = None,
        include_suggestions: bool = True,
        include_patterns: bool = True,
    ) -> AIAnalysisResult | None:
        """
        Analyze an existing test failure using AI.

        Args:
            test_result: Test result to analyze
            source_code: Source code that was tested
            test_code: Test code that failed
            include_suggestions: Whether to include fix suggestions
            include_patterns: Whether to include pattern detection

        Returns:
            AI analysis result or None if analysis failed
        """
        if not self.enable_ai_analysis or not self.ai_analysis_service:
            self.logger.warning("AI analysis is disabled or not available")
            return None

        try:
            # Create a mock request for analysis
            mock_request = TestRequest(
                code=source_code or "",
                test_code=test_code or "",
                language=getattr(test_result, "language", "python"),
                security_level=getattr(test_result, "security_level", None),
                timeout=None,
                dependencies={},
            )

            # Perform AI analysis
            analysis_result = await self._analyze_test_failure(
                test_result=test_result,
                request=mock_request,
                include_code_context=True,
                include_suggestions=include_suggestions,
                include_patterns=include_patterns,
            )

            return analysis_result

        except Exception as e:
            self.logger.error(f"Analysis of existing failure failed: {str(e)}")
            return None

    async def batch_analyze_failures(
        self,
        test_results: list[TestResult],
        max_concurrent: int | None = None,
        include_suggestions: bool = True,
        include_patterns: bool = True,
    ) -> list[AIAnalysisResult | None]:
        """
        Analyze multiple test failures concurrently.

        Args:
            test_results: List of test results to analyze
            max_concurrent: Maximum concurrent analyses (uses service default if None)
            include_suggestions: Whether to include fix suggestions
            include_patterns: Whether to include pattern detection

        Returns:
            List of analysis results (None for failures that couldn't be analyzed)
        """
        if not self.enable_ai_analysis or not self.ai_analysis_service:
            self.logger.warning("AI analysis is disabled or not available")
            return cast(list[AIAnalysisResult | None], [None] * len(test_results))

        # Filter to only failed tests
        failed_tests = [(i, result) for i, result in enumerate(test_results) if not result.success]

        if not failed_tests:
            self.logger.info("No failed tests to analyze")
            return cast(list[AIAnalysisResult | None], [None] * len(test_results))

        # Use service default if max_concurrent not specified
        if max_concurrent is None:
            max_concurrent = self.max_concurrent_analyses

        # Create semaphore for concurrent analyses
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_single_failure(
            index: int, test_result: TestResult
        ) -> tuple[int, AIAnalysisResult | None]:
            """Analyze a single failure with semaphore control."""
            async with semaphore:
                try:
                    analysis_result = await self.analyze_existing_failure(
                        test_result=test_result,
                        include_suggestions=include_suggestions,
                        include_patterns=include_patterns,
                    )
                    return index, analysis_result
                except Exception as e:
                    self.logger.error(f"Failed to analyze test {test_result.test_id}: {str(e)}")
                    return index, None

        # Run analyses concurrently
        tasks = [analyze_single_failure(index, test_result) for index, test_result in failed_tests]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Create result list with proper indexing
        analysis_results: list[AIAnalysisResult | None] = cast(
            list[AIAnalysisResult | None], [None] * len(test_results)
        )

        for result in results:
            if isinstance(result, BaseException):
                self.logger.error(f"Batch analysis task failed: {str(result)}")
                continue

            # Result is a tuple of (index, analysis_result)
            index, analysis_result = result
            analysis_results[index] = analysis_result

        self.logger.info(f"Batch analysis completed for {len(failed_tests)} failures")
        return analysis_results

    async def _analyze_test_failure(
        self,
        test_result: TestResult,
        request: TestRequest,
        include_code_context: bool = True,
        include_suggestions: bool = True,
        include_patterns: bool = True,
    ) -> AIAnalysisResult | None:
        """
        Analyze a test failure using the AI analysis service.

        Args:
            test_result: Test result to analyze
            request: Original test request
            include_code_context: Whether to include code context
            include_suggestions: Whether to include fix suggestions
            include_patterns: Whether to include pattern detection

        Returns:
            AI analysis result or None if analysis failed
        """
        try:
            # Convert test result to AI analysis format
            analysis_request = self._create_analysis_request(
                test_result=test_result,
                request=request,
                include_suggestions=include_suggestions,
                include_patterns=include_patterns,
            )

            # Perform AI analysis with timeout
            if self.ai_analysis_service is None:
                self.logger.warning("AI analysis service not available")
                return None
            async with self._analysis_semaphore:
                analysis_response = await asyncio.wait_for(
                    self.ai_analysis_service.analyze_test_failure(analysis_request),
                    timeout=self.analysis_timeout,
                )

            # Return analysis result if successful
            if analysis_response.success and analysis_response.analysis_result:
                return analysis_response.analysis_result
            else:
                self.logger.warning(f"AI analysis failed: {analysis_response.error_message}")
                return None

        except TimeoutError:
            self.logger.error(f"AI analysis timed out after {self.analysis_timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"AI analysis failed: {str(e)}")
            return None

    def _create_analysis_request(
        self,
        test_result: TestResult,
        request: TestRequest,
        include_suggestions: bool = True,
        include_patterns: bool = True,
    ) -> FailureAnalysisRequest:
        """
        Create an AI analysis request from test result and request.

        Args:
            test_result: Test execution result
            request: Original test request
            include_suggestions: Whether to include fix suggestions
            include_patterns: Whether to include pattern detection

        Returns:
            AI analysis request
        """
        # Create test failure context
        context = TestFailureContext(
            test_name=f"test_{test_result.test_id}",
            test_file=self._create_test_file_metadata(request, test_result),
            framework=self._determine_test_framework(test_result),
            execution_environment=self._extract_execution_environment(test_result),
        )

        # Create test failure data
        stdout = getattr(test_result, "stdout", "")
        stderr = getattr(test_result, "stderr", "")
        failure_data = TestFailureData(
            failure_id=test_result.test_id,
            timestamp=test_result.created_at,
            raw_output=f"{stdout}\n{stderr}",
            exit_code=getattr(test_result, "exit_code", None),
            execution_time=test_result.execution_time,
            context=context,
        )

        # Create analysis request
        analysis_request = FailureAnalysisRequest(
            request_id=str(uuid.uuid4()),
            failure_data=failure_data,
            include_suggestions=include_suggestions,
            include_patterns=include_patterns,
            max_root_causes=5,
            max_suggestions=10,
            minimum_confidence=0.1,
        )

        return analysis_request

    def _create_test_file_metadata(
        self, request: TestRequest, test_result: TestResult
    ) -> TestFileMetadata | None:
        """Create test file metadata from request and result."""
        test_code = getattr(request, "test_code", None)
        if not test_code:
            return None

        return TestFileMetadata(
            file_path=f"test_{test_result.test_id}.py",
            file_size=len(test_code),
            line_count=len(test_code.split("\n")),
            last_modified=test_result.created_at,
            framework=self._determine_test_framework(test_result),
            test_count=1,  # Simplified for now
        )

    def _determine_test_framework(self, test_result: TestResult) -> TestFramework:
        """Determine the test framework used."""
        language = getattr(test_result, "language", "python")
        if language == "python":
            # Check stdout/stderr for framework indicators
            stdout = getattr(test_result, "stdout", "")
            stderr = getattr(test_result, "stderr", "")
            output = f"{stdout}\n{stderr}".lower()
            if "pytest" in output:
                return TestFramework.PYTEST
            elif "unittest" in output:
                return TestFramework.UNITTEST
            else:
                return TestFramework.PYTEST  # Default for Python
        elif language in ["javascript", "typescript"]:
            return TestFramework.JEST  # Default for JS/TS
        else:
            return TestFramework.UNKNOWN

    def _extract_execution_environment(self, test_result: TestResult) -> dict[str, Any]:
        """Extract execution environment information."""
        security_level = getattr(test_result, "security_level", None)
        environment = {
            "language": getattr(test_result, "language", "python"),
            "security_level": (security_level.value if security_level else "MEDIUM"),
            "container_name": getattr(test_result, "container_name", None),
            "execution_time": test_result.execution_time,
        }

        # Add test metadata if available
        test_metadata = getattr(test_result, "test_metadata", None)
        if test_metadata:
            environment.update(test_metadata)

        return environment

    def _integrate_analysis_results(
        self, test_result: TestResult, analysis_result: AIAnalysisResult
    ) -> TestResult:
        """
        Integrate AI analysis results into the test result.

        Args:
            test_result: Original test result
            analysis_result: AI analysis result

        Returns:
            Test result with integrated analysis
        """
        # Initialize test metadata if not present
        test_metadata = getattr(test_result, "test_metadata", None)
        if not test_metadata:
            test_metadata = {}

        # Helper: extract enum value safely (use_enum_values=True stores plain strings)
        def _enum_val(v: object) -> object:
            return v.value if hasattr(v, "value") else v

        # Add AI analysis results
        test_metadata["ai_analysis"] = {
            "analysis_id": analysis_result.analysis_id,
            "timestamp": analysis_result.timestamp.isoformat(),
            "confidence": _enum_val(analysis_result.analysis_confidence),
            "confidence_score": analysis_result.analysis_confidence_score,
            "processing_time": analysis_result.processing_time,
            "ai_model_version": analysis_result.ai_model_version,
        }

        # Add root causes
        if analysis_result.root_causes:
            test_metadata["root_causes"] = [
                {
                    "id": cause.cause_id,
                    "category": cause.category,
                    "description": cause.description,
                    "severity": _enum_val(cause.severity),
                    "confidence": _enum_val(cause.confidence),
                    "confidence_score": cause.confidence_score,
                    "evidence": cause.evidence,
                }
                for cause in analysis_result.root_causes
            ]

        # Add fix suggestions
        if analysis_result.fix_suggestions:
            test_metadata["fix_suggestions"] = [
                {
                    "id": suggestion.suggestion_id,
                    "title": suggestion.title,
                    "description": suggestion.description,
                    "fix_type": suggestion.fix_type,
                    "priority": suggestion.priority,
                    "confidence": _enum_val(suggestion.confidence),
                    "confidence_score": suggestion.confidence_score,
                    "estimated_effort": suggestion.estimated_effort,
                    "code_examples": suggestion.code_examples,
                    "files_to_modify": suggestion.files_to_modify,
                    "verification_steps": suggestion.verification_steps,
                }
                for suggestion in analysis_result.fix_suggestions
            ]

        # Add detected patterns
        if analysis_result.detected_patterns:
            test_metadata["detected_patterns"] = [
                {
                    "id": pattern.pattern_id,
                    "name": pattern.pattern_name,
                    "type": pattern.pattern_type,
                    "confidence": _enum_val(pattern.confidence),
                    "confidence_score": pattern.confidence_score,
                    "frequency": pattern.frequency,
                    "typical_causes": pattern.typical_causes,
                    "common_fixes": pattern.common_fixes,
                }
                for pattern in analysis_result.detected_patterns
            ]

        # Add assertion details if available
        if analysis_result.assertion_details:
            test_metadata["assertion_details"] = {
                "assertion_type": analysis_result.assertion_details.assertion_type,
                "expected_value": analysis_result.assertion_details.expected_value,
                "actual_value": analysis_result.assertion_details.actual_value,
                "comparison_operator": analysis_result.assertion_details.comparison_operator,
                "assertion_message": analysis_result.assertion_details.assertion_message,
                "value_difference": analysis_result.assertion_details.value_difference,
            }

        # Set test_metadata back (works with extra="allow" model)
        test_result.test_metadata = test_metadata

        return test_result

    async def get_analysis_service_health(self) -> dict[str, Any]:
        """Get health status of the AI analysis service."""
        if not self.ai_analysis_service:
            return {
                "service": "ai_analysis",
                "status": "disabled",
                "message": "AI analysis service is not enabled",
            }

        try:
            return self.ai_analysis_service.get_health_status()
        except Exception as e:
            return {
                "service": "ai_analysis",
                "status": "error",
                "message": f"Failed to get health status: {str(e)}",
            }

    def get_integration_metrics(self) -> dict[str, Any]:
        """Get integration service metrics."""
        return {
            "service": "ai_test_integration",
            "metrics": self._metrics.copy(),
            "analysis_enabled": self.enable_ai_analysis,
            "analysis_service_available": self.ai_analysis_service is not None,
            "analysis_timeout": self.analysis_timeout,
            "max_concurrent_analyses": self.max_concurrent_analyses,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check for the integration service."""
        checks: dict[str, Any] = {}
        health_status: dict[str, Any] = {
            "service": "ai_test_integration",
            "status": "healthy",
            "checks": checks,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Check test execution service
        try:
            test_execution_health = await self.test_execution_service.health_check()
            checks["test_execution"] = {
                "status": test_execution_health.get("status", "unknown"),
                "details": test_execution_health,
            }
        except Exception as e:
            checks["test_execution"] = {
                "status": "error",
                "error": str(e),
            }

        # Check AI analysis service
        try:
            ai_analysis_health = await self.get_analysis_service_health()
            checks["ai_analysis"] = {
                "status": ai_analysis_health.get("status", "unknown"),
                "details": ai_analysis_health,
            }
        except Exception as e:
            checks["ai_analysis"] = {
                "status": "error",
                "error": str(e),
            }

        # Add integration metrics
        health_status["metrics"] = self.get_integration_metrics()

        # Determine overall status
        unhealthy_checks = [
            check for check in checks.values() if check.get("status") not in ["healthy", "disabled"]
        ]

        if unhealthy_checks:
            health_status["status"] = "degraded" if len(unhealthy_checks) == 1 else "unhealthy"

        return health_status

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.ai_analysis_service:
                await self.ai_analysis_service.cleanup()

            self.logger.info("AI Test Integration Service cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")


# Factory function for dependency injection
def create_ai_test_integration_service(
    test_execution_service: Any | None = None,
    ai_analysis_service: AIFailureAnalysisService | None = None,
    enable_ai_analysis: bool = True,
    analysis_timeout: float = 60.0,
    max_concurrent_analyses: int = 3,
) -> AITestIntegrationService:
    """
    Factory function to create AI Test Integration Service.

    Args:
        test_execution_service: Existing test execution service (optional)
        ai_analysis_service: Optional AI analysis service
        enable_ai_analysis: Whether to enable AI analysis
        analysis_timeout: Timeout for AI analysis operations
        max_concurrent_analyses: Maximum concurrent analyses

    Returns:
        Configured AI Test Integration Service
    """
    return AITestIntegrationService(
        test_execution_service=test_execution_service,
        ai_analysis_service=ai_analysis_service,
        enable_ai_analysis=enable_ai_analysis,
        analysis_timeout=analysis_timeout,
        max_concurrent_analyses=max_concurrent_analyses,
    )
