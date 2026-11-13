"""Tests for health and root endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.main import app

    return TestClient(app)


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "coding-ai"
    assert "version" in data


def test_root_returns_service_info(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "coding-ai"
    assert "version" in data
    assert data["health"] == "/health"
    assert data["docs"] == "/docs"


def test_health_includes_redis_field(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "redis" in data
    assert data["redis"] in ("connected", "disconnected")
