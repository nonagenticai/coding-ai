"""
AI-powered test failure pattern detection and classification system.

This module provides intelligent pattern detection and classification for test failures,
enabling the system to learn from historical failures and improve analysis accuracy
over time.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from coding_client.llm import (
    last_failover_chain_id,
    last_inference_id,
    llm_request_context,
)

from models.ai_test_failure_models import (
    ConfidenceLevel,
    ErrorDetails,
    TestFailureData,
    TestFailurePattern,
)
from services.llm_agents import SCHEMA_TEST_FAILURE, make_agent_runner


class AIPatternDetector:
    """
    AI-powered pattern detection and classification system.

    This system analyzes test failures to identify patterns and learn from
    historical data.
    """

    def __init__(self, model_name: str | None = None) -> None:
        # ``model_name`` retained for backwards compatibility; provider is now
        # selected server-side by inference-router. The value is stored only
        # for log/diagnostic continuity.
        self.model_name = model_name or SCHEMA_TEST_FAILURE
        self.logger = logging.getLogger(__name__)

        self._init_pattern_agents()

        self._known_patterns: dict[str, TestFailurePattern] = {}
        self._pattern_history: list[dict[str, Any]] = []
        self._pattern_signatures: dict[str, str] = {}

        self._similarity_threshold = 0.7
        self._pattern_confidence_threshold = 0.5
        self._max_pattern_age_days = 30

        self._pattern_accuracy: dict[str, float] = {}
        self._pattern_usage_count: dict[str, int] = defaultdict(int)

    def _init_pattern_agents(self) -> None:
        """Initialize specialized AI agents for pattern analysis."""

        self.pattern_extractor = make_agent_runner(
            worker_name="cai-pattern-detector-pattern-extractor",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=list[TestFailurePattern],
            system_prompt="""
            You are an expert in pattern recognition and software testing.
            Your task is to identify recurring patterns in test failures.
            """,
        )

        # str/float outputs use the shim's unstructured-text fallback; the
        # caller already coerces the response.
        self.pattern_classifier = make_agent_runner(
            worker_name="cai-pattern-detector-pattern-classifier",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=str,
            system_prompt="""
            You are an expert in software testing and failure classification.
            Your task is to classify test failure patterns into appropriate categories.
            """,
        )

        self.similarity_analyzer = make_agent_runner(
            worker_name="cai-pattern-detector-similarity-analyzer",
            schema_id=SCHEMA_TEST_FAILURE,
            output_type=float,
            system_prompt="""
            You are an expert in pattern similarity analysis.
            Your task is to determine how similar two test failure patterns are.
            Return a similarity score from 0.0 to 1.0.
            """,
        )

    async def detect_patterns(
        self,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
        historical_context: list[dict[str, Any]] | None = None,
    ) -> list[TestFailurePattern]:
        """Detect patterns in a test failure."""
        try:
            self.logger.info(f"Detecting patterns for failure {failure_data.failure_id}")

            with llm_request_context(schema_id=SCHEMA_TEST_FAILURE):
                extracted_patterns = await self._extract_patterns_ai(failure_data, error_details)
                matched_patterns = await self._match_known_patterns(failure_data, error_details)

            all_patterns = extracted_patterns + matched_patterns
            deduplicated_patterns = self._deduplicate_patterns(all_patterns)

            await self._learn_from_failure(failure_data, error_details, deduplicated_patterns)

            sorted_patterns = sorted(
                deduplicated_patterns, key=lambda p: p.confidence_score, reverse=True
            )

            self.logger.info(
                "Detected %d patterns for failure %s (inference_id=%s, failover_chain_id=%s)",
                len(sorted_patterns),
                failure_data.failure_id,
                last_inference_id(),
                last_failover_chain_id(),
            )
            return sorted_patterns

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}")
            return []

    async def _extract_patterns_ai(
        self, failure_data: TestFailureData, error_details: ErrorDetails
    ) -> list[TestFailurePattern]:
        """Extract patterns using AI analysis."""
        try:
            analysis_input = self._prepare_pattern_input(failure_data, error_details)
            result = await self.pattern_extractor.run(
                f"Extract patterns from this test failure:\n\n{analysis_input}"
            )

            enhanced_patterns = []
            for pattern in result.output:
                enhanced_pattern = await self._enhance_pattern(pattern, failure_data, error_details)
                enhanced_patterns.append(enhanced_pattern)

            return enhanced_patterns

        except Exception as e:
            self.logger.error(f"AI pattern extraction failed: {str(e)}")
            return []

    async def _match_known_patterns(
        self, failure_data: TestFailureData, error_details: ErrorDetails
    ) -> list[TestFailurePattern]:
        """Match failure against known patterns."""
        matched_patterns = []

        try:
            failure_signature = self._create_failure_signature(failure_data, error_details)

            if failure_signature in self._pattern_signatures:
                pattern_id = self._pattern_signatures[failure_signature]
                if pattern_id in self._known_patterns:
                    pattern = self._known_patterns[pattern_id]
                    pattern.frequency += 1
                    matched_patterns.append(pattern)

            for pattern_id, known_pattern in self._known_patterns.items():
                if pattern_id in [p.pattern_id for p in matched_patterns]:
                    continue

                similarity = await self._calculate_pattern_similarity(
                    failure_data, error_details, known_pattern
                )

                if similarity >= self._similarity_threshold:
                    similar_pattern = TestFailurePattern(
                        pattern_id=f"similar_{pattern_id}_{str(uuid.uuid4())[:8]}",
                        pattern_name=f"Similar to {known_pattern.pattern_name}",
                        pattern_signature=failure_signature,
                        pattern_type=known_pattern.pattern_type,
                        confidence=self._score_to_confidence(similarity * 0.8),
                        confidence_score=similarity * 0.8,
                        associated_failures=[failure_data.failure_id],
                        typical_causes=known_pattern.typical_causes,
                        common_fixes=known_pattern.common_fixes,
                    )
                    matched_patterns.append(similar_pattern)

            return matched_patterns

        except Exception as e:
            self.logger.error(f"Pattern matching failed: {str(e)}")
            return []

    async def _calculate_pattern_similarity(
        self,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
        known_pattern: TestFailurePattern,
    ) -> float:
        """Calculate similarity between failure and known pattern."""
        try:
            comparison_input = self._prepare_similarity_input(
                failure_data, error_details, known_pattern
            )
            result = await self.similarity_analyzer.run(
                f"Calculate similarity between these patterns:\n\n{comparison_input}"
            )
            return float(result.output)
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {str(e)}")
            return 0.0

    async def _enhance_pattern(
        self,
        pattern: TestFailurePattern,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
    ) -> TestFailurePattern:
        """Enhance a pattern with additional information."""
        try:
            pattern.pattern_id = f"pattern_{str(uuid.uuid4())[:8]}"

            if not pattern.associated_failures:
                pattern.associated_failures = []
            pattern.associated_failures.append(failure_data.failure_id)

            if not pattern.pattern_type or pattern.pattern_type == "unknown":
                pattern.pattern_type = await self._classify_pattern_type(pattern, error_details)  # ty: ignore[invalid-assignment]

            pattern.pattern_signature = self._create_pattern_signature(
                pattern, failure_data, error_details
            )

            return pattern

        except Exception as e:
            self.logger.error(f"Pattern enhancement failed: {str(e)}")
            return pattern

    async def _classify_pattern_type(
        self, pattern: TestFailurePattern, error_details: ErrorDetails
    ) -> str:
        """Classify the type of a pattern."""
        try:
            classification_input = self._prepare_classification_input(pattern, error_details)
            result = await self.pattern_classifier.run(
                f"Classify this pattern type:\n\n{classification_input}"
            )
            return result.output
        except Exception as e:
            self.logger.error(f"Pattern classification failed: {str(e)}")
            return "unknown"

    async def _learn_from_failure(
        self,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
        patterns: list[TestFailurePattern],
    ) -> None:
        """Learn from a failure and update pattern knowledge."""
        try:
            history_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "failure_id": failure_data.failure_id,
                "error_type": error_details.error_type,
                "error_message": error_details.error_message,
                "patterns": [p.pattern_id for p in patterns],
            }
            self._pattern_history.append(history_entry)

            for pattern in patterns:
                if pattern.pattern_id not in self._known_patterns:
                    self._known_patterns[pattern.pattern_id] = pattern
                    self._pattern_signatures[pattern.pattern_signature] = pattern.pattern_id
                else:
                    existing_pattern = self._known_patterns[pattern.pattern_id]
                    existing_pattern.frequency += 1

                    if not existing_pattern.associated_failures:
                        existing_pattern.associated_failures = []
                    if failure_data.failure_id not in existing_pattern.associated_failures:
                        existing_pattern.associated_failures.append(failure_data.failure_id)

                self._pattern_usage_count[pattern.pattern_id] += 1

            self._cleanup_old_patterns()

        except Exception as e:
            self.logger.error(f"Learning from failure failed: {str(e)}")

    def _deduplicate_patterns(self, patterns: list[TestFailurePattern]) -> list[TestFailurePattern]:
        seen_signatures = set()
        unique_patterns = []
        for pattern in patterns:
            if pattern.pattern_signature not in seen_signatures:
                seen_signatures.add(pattern.pattern_signature)
                unique_patterns.append(pattern)
        return unique_patterns

    def _prepare_pattern_input(
        self, failure_data: TestFailureData, error_details: ErrorDetails
    ) -> str:
        parts = [
            f"Test Name: {failure_data.context.test_name}",
            f"Framework: {failure_data.context.framework}",
            f"Error Type: {error_details.error_type}",
            f"Error Message: {error_details.error_message}",
        ]
        if error_details.file_path:
            parts.append(f"File: {error_details.file_path}")
        if error_details.stack_trace:
            parts.append("Stack Trace:")
            parts.extend(error_details.stack_trace[:10])
        parts.append(f"Raw Output (first 1000 chars):\n{failure_data.raw_output[:1000]}")
        return "\n".join(parts)

    def _prepare_similarity_input(
        self,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
        known_pattern: TestFailurePattern,
    ) -> str:
        parts = [
            "=== Current Failure ===",
            f"Error Type: {error_details.error_type}",
            f"Error Message: {error_details.error_message}",
            f"Test Name: {failure_data.context.test_name}",
            "",
            "=== Known Pattern ===",
            f"Pattern Name: {known_pattern.pattern_name}",
            f"Pattern Type: {known_pattern.pattern_type}",
            f"Pattern Signature: {known_pattern.pattern_signature}",
        ]
        if known_pattern.typical_causes:
            parts.append(f"Typical Causes: {', '.join(known_pattern.typical_causes)}")
        return "\n".join(parts)

    def _prepare_classification_input(
        self, pattern: TestFailurePattern, error_details: ErrorDetails
    ) -> str:
        parts = [
            f"Pattern Name: {pattern.pattern_name}",
            f"Pattern Signature: {pattern.pattern_signature}",
            f"Error Type: {error_details.error_type}",
            f"Error Message: {error_details.error_message}",
        ]
        if error_details.stack_trace:
            parts.append("Stack Trace Sample:")
            parts.extend(error_details.stack_trace[:5])
        return "\n".join(parts)

    def _create_failure_signature(
        self, failure_data: TestFailureData, error_details: ErrorDetails
    ) -> str:
        signature_parts = [
            error_details.error_type,
            self._normalize_error_message(error_details.error_message),
            failure_data.context.framework,
        ]
        if error_details.file_path:
            filename = error_details.file_path.split("/")[-1]
            signature_parts.append(filename)
        return "|".join(signature_parts)

    def _create_pattern_signature(
        self,
        pattern: TestFailurePattern,
        failure_data: TestFailureData,
        error_details: ErrorDetails,
    ) -> str:
        if pattern.pattern_signature:
            return pattern.pattern_signature
        signature_parts = [
            pattern.pattern_type,
            self._normalize_pattern_name(pattern.pattern_name),
            error_details.error_type,
        ]
        return "|".join(signature_parts)

    def _normalize_error_message(self, message: str) -> str:
        normalized = re.sub(r"\d+", "N", message)
        normalized = re.sub(r'["\'].*?["\']', "STR", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()[:100]

    def _normalize_pattern_name(self, name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())[:50]

    def _score_to_confidence(self, score: float) -> ConfidenceLevel:
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _cleanup_old_patterns(self) -> None:
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=self._max_pattern_age_days)

            self._pattern_history = [
                entry
                for entry in self._pattern_history
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
            ]

            active_pattern_ids = set()
            for entry in self._pattern_history:
                active_pattern_ids.update(entry["patterns"])

            patterns_to_remove = [
                pattern_id
                for pattern_id in self._known_patterns.keys()
                if pattern_id not in active_pattern_ids
            ]

            for pattern_id in patterns_to_remove:
                pattern = self._known_patterns.pop(pattern_id, None)
                if pattern and pattern.pattern_signature in self._pattern_signatures:
                    del self._pattern_signatures[pattern.pattern_signature]
                if pattern_id in self._pattern_usage_count:
                    del self._pattern_usage_count[pattern_id]
                if pattern_id in self._pattern_accuracy:
                    del self._pattern_accuracy[pattern_id]

            if patterns_to_remove:
                self.logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")

        except Exception as e:
            self.logger.error(f"Pattern cleanup failed: {str(e)}")

    def get_pattern_statistics(self) -> dict[str, Any]:
        return {
            "total_known_patterns": len(self._known_patterns),
            "total_history_entries": len(self._pattern_history),
            "pattern_types": list(set(p.pattern_type for p in self._known_patterns.values())),
            "most_common_patterns": sorted(
                self._pattern_usage_count.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "pattern_accuracy": dict(self._pattern_accuracy),
            "similarity_threshold": self._similarity_threshold,
            "confidence_threshold": self._pattern_confidence_threshold,
        }

    def export_patterns(self) -> dict[str, Any]:
        return {
            "patterns": {
                pattern_id: {
                    "pattern_name": pattern.pattern_name,
                    "pattern_signature": pattern.pattern_signature,
                    "pattern_type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "confidence_score": pattern.confidence_score,
                    "typical_causes": pattern.typical_causes,
                    "common_fixes": pattern.common_fixes,
                }
                for pattern_id, pattern in self._known_patterns.items()
            },
            "history": self._pattern_history,
            "usage_count": dict(self._pattern_usage_count),
            "accuracy": dict(self._pattern_accuracy),
        }

    def import_patterns(self, pattern_data: dict[str, Any]) -> None:
        try:
            patterns_data = pattern_data.get("patterns", {})
            for pattern_id, pattern_info in patterns_data.items():
                pattern = TestFailurePattern(
                    pattern_id=pattern_id,
                    pattern_name=pattern_info["pattern_name"],
                    pattern_signature=pattern_info["pattern_signature"],
                    pattern_type=pattern_info["pattern_type"],
                    frequency=pattern_info.get("frequency", 1),
                    confidence=self._score_to_confidence(pattern_info.get("confidence_score", 0.5)),
                    confidence_score=pattern_info.get("confidence_score", 0.5),
                    typical_causes=pattern_info.get("typical_causes"),
                    common_fixes=pattern_info.get("common_fixes"),
                )
                self._known_patterns[pattern_id] = pattern
                self._pattern_signatures[pattern.pattern_signature] = pattern_id

            self._pattern_history = pattern_data.get("history", [])
            self._pattern_usage_count.update(pattern_data.get("usage_count", {}))
            self._pattern_accuracy.update(pattern_data.get("accuracy", {}))

            self.logger.info(f"Imported {len(patterns_data)} patterns")

        except Exception as e:
            self.logger.error(f"Pattern import failed: {str(e)}")
