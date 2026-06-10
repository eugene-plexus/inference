"""Tests for the inference domain routes (v0.3 skeleton).

Engine-dependent endpoints (chat/completions, create endpoint, load
endpoint) return 501 (serving engine not implemented); the
engine-independent endpoints (models list, endpoints list) return real
empty-shape responses.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_list_models_returns_empty_openai_shape(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_list_endpoints_returns_empty(client: TestClient) -> None:
    response = client.get("/v1/inference/endpoints")
    assert response.status_code == 200
    assert response.json() == {"endpoints": []}


def test_chat_completion_returns_501(client: TestClient) -> None:
    request_body = {
        "model": "some-endpoint",
        "messages": [{"role": "user", "content": "hello"}],
    }
    response = client.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 501
    body = response.json()
    assert body["component"] == "inference"
    assert "not implemented" in body["detail"].lower()


def test_create_endpoint_returns_501(client: TestClient) -> None:
    request_body = {
        "endpointId": str(uuid4()),
        "name": "my-model",
        "checkpointId": str(uuid4()),
    }
    response = client.post("/v1/inference/endpoints", json=request_body)
    assert response.status_code == 501
    body = response.json()
    assert body["component"] == "inference"
    assert "not implemented" in body["detail"].lower()


def test_load_endpoint_returns_501(client: TestClient) -> None:
    endpoint_id = uuid4()
    request_body = {"checkpointId": str(uuid4())}
    response = client.post(f"/v1/inference/endpoints/{endpoint_id}/load", json=request_body)
    assert response.status_code == 501
    body = response.json()
    assert body["component"] == "inference"
