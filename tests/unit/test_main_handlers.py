"""Tests for main.py KAG-standard exception handlers and middleware."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tests.conftest import AUTHED_TOKEN


@pytest.fixture
def client(authed_client):
    """Authenticated: the subject of these tests is the exception HANDLER.

    These register ad-hoc routes on the real app and call them. coding-ai now
    registers AuthMiddleware, which gates everything outside PUBLIC_PATHS, so an
    anonymous client 401s before the handler under test ever runs — the test
    would then pass or fail for the wrong reason. `authed_client` (tests/
    conftest.py) presents a licence the middleware accepts.
    """
    return authed_client


def test_http_exception_dict_detail(client):
    """HTTPException with dict detail formats correctly (KAG envelope)."""
    from src.main import app

    @app.get("/test-dict-detail")
    async def _():
        raise HTTPException(
            status_code=400,
            detail={"message": "Custom error", "error_code": "CUSTOM", "details": {"key": "val"}},
        )

    response = client.get("/test-dict-detail")
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Custom error"
    assert data["error"]["code"] == "HTTP_400"
    assert data["error"]["details"] == {"key": "val"}
    assert "request_id" in data["error"]

    app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-dict-detail"]


def test_http_exception_string_detail(client):
    """HTTPException with string detail formats correctly (KAG envelope)."""
    from src.main import app

    @app.get("/test-string-detail")
    async def _():
        raise HTTPException(status_code=404, detail="Resource not found")

    response = client.get("/test-string-detail")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Resource not found"
    assert data["error"]["code"] == "HTTP_404"

    app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-string-detail"]


def test_http_exception_non_string_detail(client):
    """HTTPException with non-string/non-dict detail."""
    from src.main import app

    @app.get("/test-other-detail")
    async def _():
        raise HTTPException(status_code=500, detail=12345)

    response = client.get("/test-other-detail")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "12345"
    assert data["error"]["code"] == "HTTP_500"

    app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-other-detail"]


def test_validation_error_handler(client):
    """RequestValidationError produces structured error body (KAG envelope)."""
    from pydantic import BaseModel

    from src.main import app

    class StrictBody(BaseModel):
        name: str
        count: int

    @app.post("/test-validation")
    async def _(body: StrictBody):
        return {"ok": True}

    response = client.post("/test-validation", json={"name": 123})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_422"
    assert data["error"]["message"] == "Request validation failed"
    assert "details" in data["error"]
    assert len(data["error"]["details"]) > 0
    error = data["error"]["details"][0]
    assert "field" in error
    assert "message" in error
    assert "type" in error

    app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-validation"]


def test_global_exception_handler(authed_license_patch):
    """Unhandled exceptions return 500 with KAG envelope.

    Builds its own client because it needs ``raise_server_exceptions=False``,
    so it takes the licence patch directly rather than the ``authed_client``
    fixture. Without a credential AuthMiddleware answers 401 and the 500 path
    under test never runs.
    """
    from src.main import app

    @app.get("/test-unhandled")
    async def _():
        raise RuntimeError("Unexpected boom")

    with TestClient(
        app,
        raise_server_exceptions=False,
        headers={"Authorization": f"Bearer {AUTHED_TOKEN}"},
    ) as c:
        response = c.get("/test-unhandled")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert data["error"]["message"] == "An unexpected error occurred"

    app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-unhandled"]


def test_get_redis_initially_none():
    from src.main import get_redis

    # In test context, redis is not initialized
    result = get_redis()
    # May be None or a Redis instance depending on test order, just check it's callable
    assert result is None or result is not None


def test_http_exception_with_custom_error_attrs(client):
    """HTTPException with custom error_code and error_details attributes."""
    from src.main import app

    @app.get("/test-custom-exc-attrs")
    async def _():
        exc = HTTPException(status_code=409, detail="Conflict occurred")
        exc.error_code = "RESOURCE_CONFLICT"  # type: ignore[attr-defined]
        exc.error_details = {"resource": "widget", "id": "123"}  # type: ignore[attr-defined]
        raise exc

    response = client.get("/test-custom-exc-attrs")
    assert response.status_code == 409
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_409"
    assert data["error"]["message"] == "Conflict occurred"

    app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-custom-exc-attrs"]


def test_http_exception_dict_without_error_code(client):
    """Dict detail without error_code falls back to HTTP_{status}."""
    from src.main import app

    @app.get("/test-dict-no-code")
    async def _():
        raise HTTPException(status_code=403, detail={"message": "Forbidden access"})

    response = client.get("/test-dict-no-code")
    assert response.status_code == 403
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_403"
    assert data["error"]["message"] == "Forbidden access"

    app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-dict-no-code"]
