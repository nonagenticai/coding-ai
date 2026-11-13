"""Tests for the production-validator on coding-ai Settings.

Mirrors the canonical pattern in inference-router/src/settings.py and the
sibling PRs (coding-testing#3, coding-analysis#3, coding-gateway#5,
coding-security#3, coding-github#3).
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from src.config.settings import Settings, get_settings

_ALL_REQUIRED = {
    "REDIS_URL": "redis://r:6379/0",
    "CODING_CONTAINERS_URL": "http://coding-containers:8691",
    "INFERENCE_ROUTER_URL": "http://inference-router:8150/api",
    "LICENSE_INTROSPECT_URL": "http://licenses:8544/license/introspect",
}


def _clean_env() -> dict[str, str]:
    """Build a minimal env that strips all the autouse-injected defaults."""
    env = {k: v for k, v in os.environ.items() if k not in _ALL_REQUIRED}
    return env


def test_empty_defaults_when_dev_env():
    """In non-production, missing required URLs are allowed (default to '')."""
    env = _clean_env()
    env["ENVIRONMENT"] = "dev"
    with mock.patch.dict(os.environ, env, clear=True):
        s = Settings()
        assert s.redis_url == ""
        assert s.coding_containers_url == ""
        assert s.inference_router_url == ""
        assert s.license_introspect_url == ""


def test_production_validator_raises_when_required_url_missing():
    """In production, ``model_post_init`` raises if any required URL is empty."""
    env = _clean_env()
    env["ENVIRONMENT"] = "production"
    with mock.patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError) as exc_info:
            Settings()
        msg = str(exc_info.value)
        assert "missing required fields in production" in msg
        for field in (
            "redis_url",
            "coding_containers_url",
            "inference_router_url",
            "license_introspect_url",
        ):
            assert field in msg


def test_production_validator_passes_when_all_required_set():
    """In production, all required URLs set → Settings constructs cleanly."""
    env = _clean_env()
    env["ENVIRONMENT"] = "production"
    env.update(_ALL_REQUIRED)
    with mock.patch.dict(os.environ, env, clear=True):
        s = Settings()
        assert s.redis_url == _ALL_REQUIRED["REDIS_URL"]
        assert s.coding_containers_url == _ALL_REQUIRED["CODING_CONTAINERS_URL"]
        assert s.license_introspect_url == _ALL_REQUIRED["LICENSE_INTROSPECT_URL"]


def test_production_validator_partial_missing_lists_only_missing():
    """Only the empty fields appear in the error message."""
    env = _clean_env()
    env["ENVIRONMENT"] = "production"
    env.update(_ALL_REQUIRED)
    del env["CODING_CONTAINERS_URL"]
    del env["INFERENCE_ROUTER_URL"]
    with mock.patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError) as exc_info:
            Settings()
        msg = str(exc_info.value)
        assert "coding_containers_url" in msg
        assert "inference_router_url" in msg
        assert "redis_url" not in msg


def test_get_settings_is_cached():
    """``get_settings()`` returns the same instance on repeated calls."""
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
