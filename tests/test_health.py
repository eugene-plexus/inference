"""Tests for /healthz.

The engine builds at startup (torch-free construction), so a normally-started
inference component reports `ok`. The degraded path is covered by
test_safe_mode.py (safe mode leaves the engine unbuilt on purpose).
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_is_reachable_and_well_formed(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "inference"
    assert "version" in body


def test_healthz_reports_ok_when_engine_built(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["safeMode"] is False
