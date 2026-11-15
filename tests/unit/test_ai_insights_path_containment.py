"""Path containment for the AI insights endpoint (py/path-injection fix).

``POST /api/ai/insights/code/analyze`` accepts ``file_paths`` in the request
body. A raw ``Path(...).exists()`` let a caller probe for — and hand the
analysis engine — any file this pod could read. These tests pin the contract
that replaced it: realpath-normalise, then require the result to start with the
configured root PLUS ``os.sep``, and return only the normalised value.

The root has NO fallback. An unset root refuses the request; it never falls
back to the CWD, ``/workspace`` or ``/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from services.ai_insights.base import CodebaseInsights

ANALYZE_URL = "/api/ai/insights/code/analyze"


def _reset_router_settings() -> None:
    """Clear the cache the ROUTER reads.

    ``src/routers/ai_insights.py`` imports ``config.settings.get_settings``;
    conftest's autouse reset clears ``src.config.settings.get_settings``, which
    is a different module object with its own lru_cache. ``main`` also caches
    an instance at import time, so the cache must be cleared after every env
    change for the guard to observe it.
    """
    from config.settings import get_settings

    get_settings.cache_clear()


@pytest.fixture
def analysis_root(tmp_path, monkeypatch):
    """A configured analysis root; every dependency below lives under it."""
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setenv("AI_INSIGHTS_WORKSPACE_ROOT", str(root))
    _reset_router_settings()
    return root


def _write(path: Path, text: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _analyse(paths: list[str]) -> list[str]:
    from routers.ai_insights import _validate_file_paths

    return _validate_file_paths(paths)


# ---------------------------------------------------------------------------
# Acceptance and normalisation
# ---------------------------------------------------------------------------
def test_inside_root_is_accepted_and_returned_normalised(analysis_root):
    target = _write(analysis_root / "src" / "main.py")
    # A ".." that stays inside the root is legitimate; the returned value must
    # be the realpath, because that is what the engine opens.
    requested = str(analysis_root / "src" / "nested" / ".." / "main.py")
    assert _analyse([requested]) == [os.path.realpath(target)]


def test_root_itself_is_not_a_valid_file_target(analysis_root):
    with pytest.raises(HTTPException) as exc_info:
        _analyse([str(analysis_root)])
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# The four escapes the alert class is about
# ---------------------------------------------------------------------------
def test_file_outside_root_is_refused_even_when_it_exists(analysis_root, tmp_path):
    outside = _write(tmp_path / "outside" / "secret.py")
    inside = _write(analysis_root / "ok.py")

    result = _analyse([str(outside), str(inside)])

    assert result == [os.path.realpath(inside)]


def test_dotdot_traversal_is_refused(analysis_root):
    outside = _write(analysis_root.parent / "escaped.py")
    assert os.path.isfile(outside)  # the traversal target really exists

    with pytest.raises(HTTPException) as exc_info:
        _analyse([str(analysis_root / ".." / outside.name)])

    assert exc_info.value.status_code == 404


def test_sibling_prefix_is_refused(analysis_root):
    """``<root>-evil`` is a sibling, not a child: string prefix alone is wrong."""
    sibling = analysis_root.parent / f"{analysis_root.name}-evil"
    secret = _write(sibling / "secret.py")
    assert str(secret).startswith(str(analysis_root))  # the trap: a naive prefix passes

    inside = _write(analysis_root / "ok.py")
    assert _analyse([str(secret), str(inside)]) == [os.path.realpath(inside)]


def test_symlink_escape_is_refused(analysis_root, tmp_path):
    outside = _write(tmp_path / "outside" / "target.py")
    link = analysis_root / "link.py"
    link.symlink_to(outside)

    inside = _write(analysis_root / "ok.py")
    assert _analyse([str(link), str(inside)]) == [os.path.realpath(inside)]


def test_null_byte_path_is_ignored(analysis_root):
    inside = _write(analysis_root / "ok.py")
    result = _analyse([f"{analysis_root}/bad\x00name.py", str(inside)])
    assert result == [os.path.realpath(inside)]


# ---------------------------------------------------------------------------
# No configured root = refuse, never fall back
# ---------------------------------------------------------------------------
def test_unset_root_refuses_the_request(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_INSIGHTS_WORKSPACE_ROOT", raising=False)
    _reset_router_settings()
    target = _write(tmp_path / "f.py")

    with pytest.raises(HTTPException) as exc_info:
        _analyse([str(target)])

    assert exc_info.value.status_code == 503
    assert "not configured" in exc_info.value.detail["message"]


def test_filesystem_root_is_not_a_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_INSIGHTS_WORKSPACE_ROOT", os.sep)
    _reset_router_settings()
    target = _write(tmp_path / "f.py")

    with pytest.raises(HTTPException) as exc_info:
        _analyse([str(target)])

    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Endpoint wiring: the guarded value is what later operations use
# ---------------------------------------------------------------------------
def _engine_stub() -> MagicMock:
    engine = MagicMock()
    engine.generate_insights = AsyncMock(
        return_value=CodebaseInsights(
            total_files_analyzed=1,
            total_entities_analyzed=0,
            analysis_time_seconds=0.0,
        )
    )
    return engine


def test_endpoint_refuses_when_root_unset(authed_client, tmp_path, monkeypatch):
    monkeypatch.delenv("AI_INSIGHTS_WORKSPACE_ROOT", raising=False)
    _reset_router_settings()
    target = _write(tmp_path / "f.py")

    with patch("routers.ai_insights._validate_container", new=AsyncMock()):
        response = authed_client.post(
            ANALYZE_URL,
            json={"container_name": "c1", "file_paths": [str(target)]},
        )

    assert response.status_code == 503


def test_endpoint_passes_only_normalised_in_root_paths_to_the_engine(
    authed_client, analysis_root, tmp_path
):
    inside = _write(analysis_root / "src" / "main.py")
    outside = _write(tmp_path / "outside" / "secret.py")
    engine = _engine_stub()

    with (
        patch("routers.ai_insights._validate_container", new=AsyncMock()),
        patch("routers.ai_insights.get_ai_insights_engine", return_value=engine),
    ):
        response = authed_client.post(
            ANALYZE_URL,
            json={
                "container_name": "c1",
                "file_paths": [str(inside), str(outside)],
            },
        )

    assert response.status_code == 200
    handed_to_engine = engine.generate_insights.call_args.kwargs["file_paths"]
    assert handed_to_engine == [os.path.realpath(inside)]
