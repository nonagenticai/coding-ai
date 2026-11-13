"""
Utilities and helpers module for intelligent testing.
Contains shared utilities, validation functions, and helper classes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

# TODO: IntelligentTestExecutionRequest, TestLanguage, TestType - service not yet extracted
# from services.test_execution.intelligent_test_orchestrator import (
#     IntelligentTestExecutionRequest,
#     TestLanguage,
#     TestType,
# )

# Import from metrics module to avoid duplication

logger = logging.getLogger(__name__)


class TestRequestValidator:
    """Validator for intelligent test requests."""

    SUPPORTED_LANGUAGES = [
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "rust",
        "csharp",
        "cpp",
        "nodejs",
    ]

    SUPPORTED_TEST_TYPES = [
        "unit",
        "integration",
        "e2e",
        "performance",
        "security",
        "benchmark",
    ]

    @classmethod
    def validate_language(cls, language: str) -> str:
        """Validate and normalize programming language."""
        normalized = language.lower().strip()
        if normalized not in cls.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        return normalized

    @classmethod
    def validate_test_type(cls, test_type: str | None) -> str:
        """Validate and normalize test type."""
        if not test_type:
            return "unit"
        normalized = test_type.lower().strip()
        if normalized not in cls.SUPPORTED_TEST_TYPES:
            logger.warning(f"Unknown test type: {test_type}, defaulting to 'unit'")
            return "unit"
        return normalized

    @staticmethod
    def validate_timeout(timeout: int) -> int:
        """Validate timeout value (30-3600 seconds)."""
        if not 30 <= timeout <= 3600:
            raise ValueError("Timeout must be between 30 and 3600 seconds")
        return timeout

    @staticmethod
    def validate_priority(priority: int) -> int:
        """Validate priority level (1-10)."""
        if not 1 <= priority <= 10:
            raise ValueError("Priority must be between 1 and 10")
        return priority

    @staticmethod
    def validate_resource_weight(weight: float) -> float:
        """Validate resource weight (0.1-10.0)."""
        if not 0.1 <= weight <= 10.0:
            raise ValueError("Resource weight must be between 0.1 and 10.0")
        return weight


class RequestHasher:
    """Utility for generating consistent hashes for test requests."""

    @staticmethod
    def hash_request(request: Any) -> str:
        """Generate a consistent hash for a test request."""
        hash_data = {
            "language": getattr(request.language, "value", str(request.language)),
            "test_type": getattr(request.test_type, "value", str(request.test_type)),
            "source_code": request.source_code,
            "test_files": request.test_files,
            "dependencies": request.dependencies,
            "security_level": getattr(request.security_level, "value", str(request.security_level)),
        }

        json_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(json_string.encode()).hexdigest()

    @staticmethod
    def hash_code(code: str, language: str) -> str:
        """Generate a hash for source code."""
        combined = f"{language}:{code}"
        return hashlib.sha256(combined.encode()).hexdigest()


def generate_execution_id(prefix: str = "exec") -> str:
    """Generate a unique execution ID."""
    timestamp = datetime.now(UTC).timestamp()
    return f"{prefix}_{timestamp}"


async def safe_async_operation(
    operation: Callable[..., Any],
    *args: Any,
    timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """Safely execute an async operation with timeout and error handling."""
    op_name = getattr(operation, "__name__", repr(operation))
    try:
        if timeout:
            return await asyncio.wait_for(operation(*args, **kwargs), timeout=timeout)
        else:
            return await operation(*args, **kwargs)
    except TimeoutError:
        logger.error(f"Operation {op_name} timed out after {timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"Operation {op_name} failed: {e}")
        return None


def sanitize_project_name(name: str) -> str:
    """Sanitize project name for safe usage."""
    if not name:
        return "unnamed_project"

    # Remove special characters and replace with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Ensure not empty and limit length
    return sanitized[:50] if sanitized else "unnamed_project"


def validate_batch_request_limits(num_tests: int, max_concurrent: int | None = None) -> None:
    """Validate batch request limits."""
    if num_tests <= 0:
        raise ValueError("Batch must contain at least one test")
    if num_tests > 100:  # Reasonable upper limit
        raise ValueError("Batch cannot exceed 100 tests")
    if max_concurrent and not 1 <= max_concurrent <= 20:
        raise ValueError("max_concurrent must be between 1 and 20")


def format_error_response(error: Exception, error_type: str = "unknown") -> dict[str, Any]:
    """Format error for API response."""
    return {
        "error": error_type,
        "message": str(error),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def calculate_estimated_batch_duration(
    test_count: int, avg_test_duration_ms: float, max_concurrent: int = 5
) -> float:
    """Calculate estimated duration for batch execution."""
    if test_count == 0:
        return 0.0

    # Simple estimation: divide tests among concurrent workers
    concurrent_batches = (test_count + max_concurrent - 1) // max_concurrent
    return concurrent_batches * avg_test_duration_ms


def is_valid_code_content(code: str) -> bool:
    """Basic validation for code content."""
    if not code or not code.strip():
        return False
    if len(code) > 1_000_000:  # 1MB limit
        return False
    return True
