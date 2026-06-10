"""Tests for /healthz.

The v0.3 inference skeleton ships no serving engine — `app.state.engine`
is None and no model is loaded — so honest health is `degraded` with an
explanatory `engine_error`. When the serving engine lands, a
successfully-built engine flips this to `ok`.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_is_reachable_and_well_formed(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "inference"
    assert "version" in body


def test_healthz_reports_degraded_in_skeleton(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    # No serving engine yet -> degraded, with an explanatory detail.
    assert body["status"] == "degraded"
    assert body["safeMode"] is False
    assert "engine" in body["details"]["engine_error"]
