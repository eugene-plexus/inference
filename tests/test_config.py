"""Tests for the config protocol endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_config_schema_lists_inference_fields(client: TestClient) -> None:
    response = client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "inference"
    keys = {f["key"] for f in body["fields"]}
    # `port` is not here — owned by the watchdog topology via
    # EUGENE_PLEXUS_INF_BIND_PORT.
    assert keys == {
        "modelsDir",
        "defaultTemperature",
        "defaultTopP",
        "defaultMaxTokens",
        "logLevel",
    }


def test_get_config_returns_defaults(client: TestClient) -> None:
    response = client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert "port" not in body
    assert body["logLevel"] == "INFO"
    assert body["defaultTemperature"] == 0.7
    assert body["defaultTopP"] == 1.0
    assert body["defaultMaxTokens"] == 512


def test_patch_config_validates_per_field(client: TestClient) -> None:
    response = client.patch(
        "/v1/config",
        json={
            "defaultTemperature": 1.2,  # valid number
            "logLevel": "DEBUG",  # valid enum, requiresRestart
            "logLevel_typo": "DEBUG",  # unknown field
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["applied"]) == {"defaultTemperature", "logLevel"}
    rejected = {r["key"] for r in body["rejected"]}
    assert rejected == {"logLevel_typo"}
    # logLevel is requiresRestart
    assert body["requiresRestart"] is True
    assert "logLevel" in body["pendingRestart"]


def test_patch_config_rejects_out_of_range_number(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"defaultTemperature": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "defaultTemperature"


def test_patch_config_rejects_bad_enum(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"logLevel": "VERBOSE"})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "logLevel"


def test_patch_config_rejects_unknown_field(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"madeUpKey": "anything"})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "madeUpKey"
    assert "unknown field" in body["rejected"][0]["message"]


def test_config_test_succeeds_when_models_dir_readable(client: TestClient) -> None:
    response = client.post("/v1/config/test", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "inference"
    assert body["ok"] is True
    assert "latencyMs" in body
