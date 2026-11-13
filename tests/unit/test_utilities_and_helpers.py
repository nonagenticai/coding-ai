"""Tests for utilities_and_helpers module."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from routers.ai_test_insights.utilities_and_helpers import (
    RequestHasher,
    TestRequestValidator,
    calculate_estimated_batch_duration,
    format_error_response,
    generate_execution_id,
    is_valid_code_content,
    safe_async_operation,
    sanitize_project_name,
    validate_batch_request_limits,
)


# ---------------------------------------------------------------------------
# TestRequestValidator
# ---------------------------------------------------------------------------
class TestValidateLanguage:
    def test_supported_language(self):
        assert TestRequestValidator.validate_language("Python") == "python"

    def test_uppercase(self):
        assert TestRequestValidator.validate_language("JAVA") == "java"

    def test_whitespace(self):
        assert TestRequestValidator.validate_language("  go  ") == "go"

    def test_unsupported_language(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            TestRequestValidator.validate_language("cobol")

    def test_all_supported(self):
        for lang in TestRequestValidator.SUPPORTED_LANGUAGES:
            assert TestRequestValidator.validate_language(lang) == lang


class TestValidateTestType:
    def test_none_defaults_to_unit(self):
        assert TestRequestValidator.validate_test_type(None) == "unit"

    def test_empty_defaults_to_unit(self):
        assert TestRequestValidator.validate_test_type("") == "unit"

    def test_supported_type(self):
        assert TestRequestValidator.validate_test_type("integration") == "integration"

    def test_uppercase(self):
        assert TestRequestValidator.validate_test_type("E2E") == "e2e"

    def test_unknown_defaults_to_unit(self):
        assert TestRequestValidator.validate_test_type("chaos") == "unit"


class TestValidateTimeout:
    def test_valid_timeout(self):
        assert TestRequestValidator.validate_timeout(60) == 60

    def test_min_boundary(self):
        assert TestRequestValidator.validate_timeout(30) == 30

    def test_max_boundary(self):
        assert TestRequestValidator.validate_timeout(3600) == 3600

    def test_too_low(self):
        with pytest.raises(ValueError, match="Timeout must be between"):
            TestRequestValidator.validate_timeout(10)

    def test_too_high(self):
        with pytest.raises(ValueError, match="Timeout must be between"):
            TestRequestValidator.validate_timeout(5000)


class TestValidatePriority:
    def test_valid(self):
        assert TestRequestValidator.validate_priority(5) == 5

    def test_boundaries(self):
        assert TestRequestValidator.validate_priority(1) == 1
        assert TestRequestValidator.validate_priority(10) == 10

    def test_too_low(self):
        with pytest.raises(ValueError, match="Priority must be between"):
            TestRequestValidator.validate_priority(0)

    def test_too_high(self):
        with pytest.raises(ValueError, match="Priority must be between"):
            TestRequestValidator.validate_priority(11)


class TestValidateResourceWeight:
    def test_valid(self):
        assert TestRequestValidator.validate_resource_weight(1.0) == 1.0

    def test_boundaries(self):
        assert TestRequestValidator.validate_resource_weight(0.1) == 0.1
        assert TestRequestValidator.validate_resource_weight(10.0) == 10.0

    def test_too_low(self):
        with pytest.raises(ValueError, match="Resource weight must be between"):
            TestRequestValidator.validate_resource_weight(0.01)

    def test_too_high(self):
        with pytest.raises(ValueError, match="Resource weight must be between"):
            TestRequestValidator.validate_resource_weight(11.0)


# ---------------------------------------------------------------------------
# RequestHasher
# ---------------------------------------------------------------------------
class TestRequestHasher:
    def test_hash_request_deterministic(self):
        req = SimpleNamespace(
            language=SimpleNamespace(value="python"),
            test_type=SimpleNamespace(value="unit"),
            source_code="print('hello')",
            test_files=["test.py"],
            dependencies={},
            security_level=SimpleNamespace(value="medium"),
        )
        h1 = RequestHasher.hash_request(req)
        h2 = RequestHasher.hash_request(req)
        assert h1 == h2
        assert len(h1) == 64  # sha256

    def test_hash_request_different_inputs(self):
        req1 = SimpleNamespace(
            language="python",
            test_type="unit",
            source_code="code1",
            test_files=[],
            dependencies={},
            security_level="medium",
        )
        req2 = SimpleNamespace(
            language="python",
            test_type="unit",
            source_code="code2",
            test_files=[],
            dependencies={},
            security_level="medium",
        )
        assert RequestHasher.hash_request(req1) != RequestHasher.hash_request(req2)

    def test_hash_code(self):
        h = RequestHasher.hash_code("print('hi')", "python")
        assert len(h) == 64
        assert RequestHasher.hash_code("print('hi')", "python") == h

    def test_hash_code_different_language(self):
        h1 = RequestHasher.hash_code("code", "python")
        h2 = RequestHasher.hash_code("code", "java")
        assert h1 != h2


# ---------------------------------------------------------------------------
# generate_execution_id
# ---------------------------------------------------------------------------
class TestGenerateExecutionId:
    def test_default_prefix(self):
        eid = generate_execution_id()
        assert eid.startswith("exec_")

    def test_custom_prefix(self):
        eid = generate_execution_id("batch")
        assert eid.startswith("batch_")


# ---------------------------------------------------------------------------
# safe_async_operation
# ---------------------------------------------------------------------------
class TestSafeAsyncOperation:
    @pytest.mark.asyncio
    async def test_success(self):
        op = AsyncMock(return_value=42)
        result = await safe_async_operation(op)
        assert result == 42

    @pytest.mark.asyncio
    async def test_timeout(self):
        async def slow_op():
            await asyncio.sleep(10)

        result = await safe_async_operation(slow_op, timeout=0.01)
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        async def failing_op():
            raise RuntimeError("fail")

        result = await safe_async_operation(failing_op)
        assert result is None

    @pytest.mark.asyncio
    async def test_with_args_and_kwargs(self):
        async def add(a, b, extra=0):
            return a + b + extra

        result = await safe_async_operation(add, 1, 2, extra=3)
        assert result == 6


# ---------------------------------------------------------------------------
# sanitize_project_name
# ---------------------------------------------------------------------------
class TestSanitizeProjectName:
    def test_normal_name(self):
        assert sanitize_project_name("my-project") == "my-project"

    def test_empty_name(self):
        assert sanitize_project_name("") == "unnamed_project"

    def test_special_characters(self):
        assert sanitize_project_name("my project!@#") == "my_project"

    def test_multiple_underscores(self):
        assert sanitize_project_name("a___b") == "a_b"

    def test_leading_trailing_underscores(self):
        result = sanitize_project_name("__project__")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_long_name_truncated(self):
        long_name = "a" * 100
        assert len(sanitize_project_name(long_name)) == 50

    def test_all_special_chars(self):
        assert sanitize_project_name("!@#$%") == "unnamed_project"


# ---------------------------------------------------------------------------
# validate_batch_request_limits
# ---------------------------------------------------------------------------
class TestValidateBatchRequestLimits:
    def test_valid(self):
        validate_batch_request_limits(10)

    def test_zero_tests(self):
        with pytest.raises(ValueError, match="at least one test"):
            validate_batch_request_limits(0)

    def test_negative_tests(self):
        with pytest.raises(ValueError, match="at least one test"):
            validate_batch_request_limits(-1)

    def test_too_many_tests(self):
        with pytest.raises(ValueError, match="cannot exceed 100"):
            validate_batch_request_limits(101)

    def test_valid_max_concurrent(self):
        validate_batch_request_limits(10, max_concurrent=5)

    def test_invalid_max_concurrent_low(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            validate_batch_request_limits(10, max_concurrent=-1)

    def test_invalid_max_concurrent_high(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            validate_batch_request_limits(10, max_concurrent=25)


# ---------------------------------------------------------------------------
# format_error_response
# ---------------------------------------------------------------------------
class TestFormatErrorResponse:
    def test_basic(self):
        err = format_error_response(ValueError("bad"), "validation")
        assert err["error"] == "validation"
        assert err["message"] == "bad"
        assert "timestamp" in err

    def test_default_error_type(self):
        err = format_error_response(RuntimeError("oops"))
        assert err["error"] == "unknown"


# ---------------------------------------------------------------------------
# calculate_estimated_batch_duration
# ---------------------------------------------------------------------------
class TestCalculateEstimatedBatchDuration:
    def test_zero_tests(self):
        assert calculate_estimated_batch_duration(0, 100.0) == 0.0

    def test_basic(self):
        # 10 tests, 100ms avg, 5 concurrent -> 2 batches * 100 = 200
        duration = calculate_estimated_batch_duration(10, 100.0, 5)
        assert duration == 200.0

    def test_not_evenly_divisible(self):
        # 7 tests, 100ms avg, 3 concurrent -> ceil(7/3)=3 batches * 100 = 300
        duration = calculate_estimated_batch_duration(7, 100.0, 3)
        assert duration == 300.0


# ---------------------------------------------------------------------------
# is_valid_code_content
# ---------------------------------------------------------------------------
class TestIsValidCodeContent:
    def test_valid(self):
        assert is_valid_code_content("print('hi')") is True

    def test_empty(self):
        assert is_valid_code_content("") is False

    def test_whitespace_only(self):
        assert is_valid_code_content("   \n\t  ") is False

    def test_too_large(self):
        assert is_valid_code_content("x" * 1_000_001) is False

    def test_at_limit(self):
        assert is_valid_code_content("x" * 1_000_000) is True
