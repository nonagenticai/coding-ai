"""
Comprehensive AI-powered test failure analysis service.

This service provides a unified interface for AI-powered test failure analysis
with robust error handling, logging, monitoring, and failover mechanisms.
It orchestrates the various AI analysis engines and provides comprehensive
error recovery and graceful degradation.

Features:
- Unified analysis interface
- Comprehensive error handling and recovery
- Detailed logging and monitoring
- Performance metrics and caching
- Graceful degradation on failures
- Rate limiting and resource management
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from models.ai_test_failure_models import (
    AIAnalysisResult,
    ConfidenceLevel,
    ErrorDetails,
    FailureAnalysisRequest,
    FailureAnalysisResponse,
    FailureType,
    TestFailureData,
)
from services.ai_fix_suggestion import AIFixSuggestionEngine
from services.ai_root_cause import AIRootCauseEngine
from services.ai_test_failure_analyzer import AITestFailureAnalyzer


class AIFailureAnalysisError(Exception):
    """Base exception for AI failure analysis errors."""

    pass


class AIAnalysisTimeoutError(AIFailureAnalysisError):
    """Raised when AI analysis times out."""

    pass


class AIModelUnavailableError(AIFailureAnalysisError):
    """Raised when AI model is unavailable."""

    pass


class AIAnalysisValidationError(AIFailureAnalysisError):
    """Raised when AI analysis results fail validation."""

    pass


class AIFailureAnalysisService:
    """
    Comprehensive AI-powered test failure analysis service.

    This service provides a robust, fault-tolerant interface for analyzing
    test failures using AI. It includes comprehensive error handling, logging,
    monitoring, and graceful degradation capabilities.
    """

    def __init__(
        self,
        model_name: str = "local-llm",
        max_concurrent_analyses: int = 5,
        analysis_timeout: float = 300.0,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize the AI failure analysis service.

        Args:
            model_name: Name of the AI model to use
            max_concurrent_analyses: Maximum concurrent analyses
            analysis_timeout: Timeout for analysis operations in seconds
            enable_caching: Whether to enable result caching
            cache_ttl: Cache time-to-live in seconds
            enable_metrics: Whether to collect performance metrics
        """
        self.model_name = model_name
        self.max_concurrent_analyses = max_concurrent_analyses
        self.analysis_timeout = analysis_timeout
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.enable_metrics = enable_metrics
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.is_available = True
        self.analyzer: AITestFailureAnalyzer | None = None
        self.root_cause_engine: AIRootCauseEngine | None = None
        self.fix_suggestion_engine: AIFixSuggestionEngine | None = None
        try:
            self._init_ai_engines()
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize AI engines: {e}. Service will operate in degraded mode."
            )
            self.is_available = False
        self._active_analyses: dict[str, datetime] = {}
        self._analysis_semaphore = asyncio.Semaphore(max_concurrent_analyses)
        self._result_cache: dict[str, tuple[AIAnalysisResult, datetime]] = {}
        self._metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "timeout_errors": 0,
            "validation_errors": 0,
            "model_errors": 0,
            "average_analysis_time": 0.0,
            "total_analysis_time": 0.0,
        }
        self._max_retries = 3
        self._retry_delay = 5.0
        self._circuit_breaker_threshold = 10
        self._circuit_breaker_timeout = 300.0
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: datetime | None = None
        self.logger.info(f"AI Failure Analysis Service initialized with model: {model_name}")

    def _init_ai_engines(self) -> None:
        """Initialize AI analysis engines with error handling."""
        try:
            self.analyzer = AITestFailureAnalyzer(self.model_name)
            self.root_cause_engine = AIRootCauseEngine(self.model_name)
            self.fix_suggestion_engine = AIFixSuggestionEngine(self.model_name)
            self.logger.info("AI engines initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI engines: {str(e)}")
            raise AIFailureAnalysisError(f"Engine initialization failed: {str(e)}")

    async def analyze_test_failure(
        self, request: FailureAnalysisRequest
    ) -> FailureAnalysisResponse:
        """
        Perform comprehensive AI-powered test failure analysis.

        Args:
            request: Test failure analysis request

        Returns:
            Complete analysis response with error handling
        """
        start_time = time.time()
        request_id = request.request_id
        try:
            self.logger.info(f"Starting analysis for request {request_id}")
            await self._validate_request(request)
            if self._is_circuit_breaker_open():
                raise AIModelUnavailableError("Circuit breaker is open due to repeated failures")
            if self.enable_caching:
                cached_result = self._get_cached_result(request)
                if cached_result:
                    self.logger.info(f"Returning cached result for request {request_id}")
                    return self._create_success_response(
                        request_id, cached_result, time.time() - start_time
                    )
            async with self._analysis_semaphore:
                analysis_result = await self._perform_analysis_with_retry(request)
            # A degraded result (LLM outage / fallback path) is never cached, so
            # a later healthy request re-runs the full analysis once the LLM
            # plane recovers instead of serving the degraded result for cache_ttl.
            if self.enable_caching and analysis_result and not self._is_degraded(analysis_result):
                self._cache_result(request, analysis_result)
            self._update_metrics(success=True, duration=time.time() - start_time)
            self._reset_circuit_breaker()
            self.logger.info(f"Analysis completed successfully for request {request_id}")
            return self._create_success_response(
                request_id, analysis_result, time.time() - start_time
            )
        except AIAnalysisTimeoutError as e:
            self.logger.error(f"Analysis timeout for request {request_id}: {str(e)}")
            self._update_metrics(
                success=False, duration=time.time() - start_time, error_type="timeout"
            )
            return self._create_error_response(
                request_id, f"Analysis timeout: {str(e)}", time.time() - start_time
            )
        except AIModelUnavailableError as e:
            self.logger.error(f"AI model unavailable for request {request_id}: {str(e)}")
            self._update_metrics(
                success=False, duration=time.time() - start_time, error_type="model"
            )
            return self._create_error_response(
                request_id, f"AI model unavailable: {str(e)}", time.time() - start_time
            )
        except AIAnalysisValidationError as e:
            self.logger.error(f"Analysis validation error for request {request_id}: {str(e)}")
            self._update_metrics(
                success=False,
                duration=time.time() - start_time,
                error_type="validation",
            )
            return self._create_error_response(
                request_id, f"Validation error: {str(e)}", time.time() - start_time
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in analysis for request {request_id}: {str(e)}")
            self._update_circuit_breaker()
            self._update_metrics(
                success=False,
                duration=time.time() - start_time,
                error_type="unexpected",
            )
            return self._create_error_response(
                request_id, f"Analysis failed: {str(e)}", time.time() - start_time
            )
        finally:
            # Clean up active analysis tracking to prevent resource leak
            self._active_analyses.pop(request_id, None)

    async def _validate_request(self, request: FailureAnalysisRequest) -> None:
        """Validate the analysis request."""
        try:
            if not request.failure_data:
                raise AIAnalysisValidationError("Missing failure data")
            if not request.failure_data.raw_output:
                raise AIAnalysisValidationError("Missing raw output in failure data")
            if not request.failure_data.context:
                raise AIAnalysisValidationError("Missing context in failure data")
            if len(request.failure_data.raw_output) > 100000:
                raise AIAnalysisValidationError("Raw output exceeds size limit (100KB)")
            if len(self._active_analyses) >= self.max_concurrent_analyses:
                raise AIAnalysisValidationError("Too many concurrent analyses")
            self._active_analyses[request.request_id] = datetime.now(UTC)
        except ValidationError as e:
            raise AIAnalysisValidationError(f"Request validation failed: {str(e)}")

    async def _perform_analysis_with_retry(
        self, request: FailureAnalysisRequest
    ) -> AIAnalysisResult:
        """Perform analysis with retry logic and timeout."""
        last_exception: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                self.logger.info(
                    f"Analysis attempt {attempt + 1}/{self._max_retries} for request {request.request_id}"
                )
                analysis_result = await asyncio.wait_for(
                    self._perform_core_analysis(request), timeout=self.analysis_timeout
                )
                if not analysis_result:
                    raise AIAnalysisValidationError("Analysis returned no results")
                return analysis_result
            except TimeoutError:
                last_exception = AIAnalysisTimeoutError(
                    f"Analysis timeout after {self.analysis_timeout}s"
                )
                self.logger.warning(
                    f"Analysis timeout on attempt {attempt + 1} for request {request.request_id}"
                )
            except AIFailureAnalysisError:
                raise
            except Exception as e:
                last_exception = AIFailureAnalysisError(str(e))
                self.logger.warning(
                    f"Analysis failed on attempt {attempt + 1} for request {request.request_id}: {str(e)}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * 2**attempt)
        if last_exception is not None:
            if isinstance(last_exception, AIAnalysisTimeoutError):
                raise last_exception
            raise AIFailureAnalysisError(
                f"Analysis failed after {self._max_retries} attempts: {str(last_exception)}"
            )
        raise AIFailureAnalysisError(
            f"Analysis failed after {self._max_retries} attempts with no exception captured"
        )

    async def _perform_core_analysis(self, request: FailureAnalysisRequest) -> AIAnalysisResult:
        """Perform the core analysis logic."""
        try:
            if self.analyzer is None:
                raise AIFailureAnalysisError("Analyzer not initialized")
            response = await self.analyzer.analyze_failure(request)
            if not response.success:
                raise AIFailureAnalysisError(f"Analysis failed: {response.error_message}")
            if not response.analysis_result:
                raise AIAnalysisValidationError("Analysis completed but returned no results")
            return response.analysis_result
        except Exception as e:
            self.logger.error(f"Core analysis failed: {str(e)}")
            return await self._perform_fallback_analysis(request)

    async def _perform_fallback_analysis(self, request: FailureAnalysisRequest) -> AIAnalysisResult:
        """Perform fallback analysis when main analysis fails."""
        try:
            self.logger.info(f"Attempting fallback analysis for request {request.request_id}")
            error_details = await self._extract_basic_error_details(request.failure_data)
            analysis_result = AIAnalysisResult(
                analysis_id=str(uuid.uuid4()),
                failure_data=request.failure_data,
                error_details=error_details,
                root_causes=[],
                fix_suggestions=[],
                detected_patterns=[],
                analysis_confidence=ConfidenceLevel.VERY_LOW,
                analysis_confidence_score=0.1,
                ai_model_version=f"fallback_{self.model_name}",
                additional_metadata={"fallback_analysis": True, "degraded": True},
            )
            self.logger.info(f"Fallback analysis completed for request {request.request_id}")
            return analysis_result
        except Exception as e:
            self.logger.error(f"Fallback analysis failed: {str(e)}")
            raise AIFailureAnalysisError(f"Both main and fallback analysis failed: {str(e)}")

    async def _extract_basic_error_details(self, failure_data: TestFailureData) -> ErrorDetails:
        """Extract basic error details without AI analysis."""
        raw_output = failure_data.raw_output
        error_type: FailureType = FailureType.RUNTIME_ERROR
        if "AssertionError" in raw_output or "AssertionError" in raw_output:
            error_type = FailureType.ASSERTION_ERROR
        elif "ImportError" in raw_output or "ModuleNotFoundError" in raw_output:
            error_type = FailureType.IMPORT_ERROR
        elif "SyntaxError" in raw_output:
            error_type = FailureType.SYNTAX_ERROR
        elif "TimeoutError" in raw_output or "timeout" in raw_output.lower():
            error_type = FailureType.TIMEOUT_ERROR
        error_message = "Unknown error"
        lines = raw_output.split("\n")
        for line in lines:
            if "Error:" in line or "Exception:" in line:
                error_message = line.strip()
                break
        return ErrorDetails(
            error_type=error_type,
            error_message=error_message,
            traceback=raw_output[:1000] if raw_output else None,
        )

    def _get_cached_result(self, request: FailureAnalysisRequest) -> AIAnalysisResult | None:
        """Get cached analysis result if available and valid."""
        try:
            cache_key = self._create_cache_key(request)
            if cache_key in self._result_cache:
                cached_result, cached_time = self._result_cache[cache_key]
                if datetime.now(UTC) - cached_time < timedelta(seconds=self.cache_ttl):
                    self._metrics["cache_hits"] += 1
                    return cached_result
                else:
                    del self._result_cache[cache_key]
            self._metrics["cache_misses"] += 1
            return None
        except Exception as e:
            self.logger.error(f"Cache lookup failed: {str(e)}")
            return None

    def _cache_result(self, request: FailureAnalysisRequest, result: AIAnalysisResult) -> None:
        """Cache analysis result."""
        try:
            cache_key = self._create_cache_key(request)
            self._result_cache[cache_key] = (result, datetime.now(UTC))
            self._cleanup_cache()
        except Exception as e:
            self.logger.error(f"Cache storage failed: {str(e)}")

    def _create_cache_key(self, request: FailureAnalysisRequest) -> str:
        """Create cache key for the request."""
        key_parts = [
            request.failure_data.failure_id,
            str(hash(request.failure_data.raw_output)),
            str(request.max_root_causes),
            str(request.max_suggestions),
            str(request.minimum_confidence),
        ]
        return "|".join(key_parts)

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        try:
            current_time = datetime.now(UTC)
            expired_keys = []
            for cache_key, (result, cached_time) in self._result_cache.items():
                if current_time - cached_time >= timedelta(seconds=self.cache_ttl):
                    expired_keys.append(cache_key)
            for key in expired_keys:
                del self._result_cache[key]
            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {str(e)}")

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker_failures < self._circuit_breaker_threshold:
            return False
        if not self._circuit_breaker_last_failure:
            return False
        time_since_last_failure = datetime.now(UTC) - self._circuit_breaker_last_failure
        return time_since_last_failure < timedelta(seconds=self._circuit_breaker_timeout)

    def _update_circuit_breaker(self) -> None:
        """Update circuit breaker state on failure."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.now(UTC)
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            self.logger.warning(
                f"Circuit breaker opened after {self._circuit_breaker_failures} failures"
            )

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state on success."""
        if self._circuit_breaker_failures > 0:
            self.logger.info("Circuit breaker reset after successful analysis")
            self._circuit_breaker_failures = 0
            self._circuit_breaker_last_failure = None

    def _update_metrics(
        self, success: bool, duration: float, error_type: str | None = None
    ) -> None:
        """Update performance metrics."""
        if not self.enable_metrics:
            return
        try:
            self._metrics["total_analyses"] += 1
            self._metrics["total_analysis_time"] += duration
            if success:
                self._metrics["successful_analyses"] += 1
            else:
                self._metrics["failed_analyses"] += 1
                if error_type:
                    metric_key = f"{error_type}_errors"
                    self._metrics[metric_key] = self._metrics.get(metric_key, 0) + 1
            self._metrics["average_analysis_time"] = (
                self._metrics["total_analysis_time"] / self._metrics["total_analyses"]
            )
        except Exception as e:
            self.logger.error(f"Metrics update failed: {str(e)}")

    @staticmethod
    def _is_degraded(analysis_result: AIAnalysisResult) -> bool:
        """True when the analysis ran on a degraded/fallback path.

        The degraded flag is carried in ``additional_metadata`` because the
        service returns the ``AIAnalysisResult`` (not the analyzer's response),
        so this is the only carrier that survives to the router.
        """
        return bool((analysis_result.additional_metadata or {}).get("degraded"))

    def _create_success_response(
        self, request_id: str, analysis_result: AIAnalysisResult, processing_time: float
    ) -> FailureAnalysisResponse:
        """Create a success response, propagating the degraded flag."""
        return FailureAnalysisResponse(
            request_id=request_id,
            success=True,
            degraded=self._is_degraded(analysis_result),
            analysis_result=analysis_result,
            processing_time=processing_time,
        )

    def _create_error_response(
        self, request_id: str, error_message: str, processing_time: float
    ) -> FailureAnalysisResponse:
        """Create an error response."""
        return FailureAnalysisResponse(
            request_id=request_id,
            success=False,
            error_message=error_message,
            processing_time=processing_time,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return self._metrics.copy()

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of the service."""
        return {
            "service_name": "AI Failure Analysis Service",
            "model_name": self.model_name,
            "active_analyses": len(self._active_analyses),
            "max_concurrent_analyses": self.max_concurrent_analyses,
            "circuit_breaker_open": self._is_circuit_breaker_open(),
            "circuit_breaker_failures": self._circuit_breaker_failures,
            "cache_size": len(self._result_cache),
            "cache_enabled": self.enable_caching,
            "metrics_enabled": self.enable_metrics,
            "metrics": self.get_metrics(),
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self._active_analyses.clear()
            self._result_cache.clear()
            self.logger.info("AI Failure Analysis Service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    @asynccontextmanager
    async def analysis_context(self, request_id: str) -> AsyncGenerator[None]:
        """Context manager for analysis operations."""
        try:
            self.logger.info(f"Starting analysis context for request {request_id}")
            yield
        finally:
            if request_id in self._active_analyses:
                del self._active_analyses[request_id]
            self.logger.info(f"Analysis context completed for request {request_id}")
