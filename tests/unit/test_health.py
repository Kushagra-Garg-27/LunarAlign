"""
SIH26166 — Backend health endpoint tests.
"""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    """GET /health should return 200 with status 'ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "sih26166-backend"
    assert "version" in data


def test_health_endpoint_method_not_allowed():
    """POST /health should return 405."""
    response = client.post("/health")
    assert response.status_code == 405
