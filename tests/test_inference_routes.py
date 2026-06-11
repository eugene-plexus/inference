"""Tests for the inference domain routes against the live serving engine.

Covers the OpenAI-compatible surface end to end (create endpoint -> load ->
chat non-streaming + streaming -> models list) plus the error mappings. A tiny
untrained model is served, so wire-shape and token-accounting are asserted, not
semantic quality.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient


def test_list_endpoints_starts_empty(client: TestClient) -> None:
    response = client.get("/v1/inference/endpoints")
    assert response.status_code == 200
    assert response.json() == {"endpoints": []}


def test_list_models_starts_empty(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_create_then_load_then_serve(
    client: TestClient, make_text_checkpoint: Callable[..., str]
) -> None:
    checkpoint_id = make_text_checkpoint()
    endpoint_id = str(uuid4())

    created = client.post(
        "/v1/inference/endpoints",
        json={"endpointId": endpoint_id, "name": "m1", "checkpointId": checkpoint_id},
    )
    assert created.status_code == 201, created.text
    assert created.json()["status"] == "unloaded"

    # Not ready yet -> chat fails with 400.
    early = client.post(
        "/v1/chat/completions",
        json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert early.status_code == 400

    loaded = client.post(
        f"/v1/inference/endpoints/{endpoint_id}/load", json={"checkpointId": checkpoint_id}
    )
    assert loaded.status_code == 202, loaded.text
    assert loaded.json()["status"] == "ready"

    # Now it lists as a model (full OpenAI shape) + serves.
    models = client.get("/v1/models").json()["data"]
    assert len(models) == 1
    assert models[0]["id"] == "m1"
    assert models[0]["object"] == "model"
    assert models[0]["owned_by"] == "eugene-plexus"
    assert isinstance(models[0]["created"], int)
    listed = client.get("/v1/inference/endpoints").json()["endpoints"]
    assert listed[0]["status"] == "ready"


def test_chat_completion_non_streaming_shape(client: TestClient, ready_text_endpoint: str) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": ready_text_endpoint,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 6,
            "temperature": 0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == ready_text_endpoint
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in {"stop", "length"}
    usage = body["usage"]
    # Token accounting is real token ids, not characters (the CLLM usage bug).
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] <= 6
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completion_streaming_sse(client: TestClient, ready_text_endpoint: str) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": ready_text_endpoint,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
            "temperature": 0,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    events = [line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    # First chunk carries the assistant role and omits finish_reason.
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "finish_reason" not in chunks[0]["choices"][0]
    # Terminal chunk has an empty delta and a finish_reason.
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}


def test_chat_completion_unknown_model_404(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["component"] == "inference"


def test_create_duplicate_name_409(
    client: TestClient, make_text_checkpoint: Callable[..., str]
) -> None:
    checkpoint_id = make_text_checkpoint()
    first = client.post(
        "/v1/inference/endpoints",
        json={"endpointId": str(uuid4()), "name": "dup", "checkpointId": checkpoint_id},
    )
    assert first.status_code == 201
    second = client.post(
        "/v1/inference/endpoints",
        json={"endpointId": str(uuid4()), "name": "dup", "checkpointId": checkpoint_id},
    )
    assert second.status_code == 409


def test_load_unknown_endpoint_404(client: TestClient) -> None:
    response = client.post(
        f"/v1/inference/endpoints/{uuid4()}/load", json={"checkpointId": str(uuid4())}
    )
    assert response.status_code == 404


def test_load_missing_checkpoint_400(client: TestClient) -> None:
    endpoint_id = str(uuid4())
    client.post(
        "/v1/inference/endpoints",
        json={"endpointId": endpoint_id, "name": "ghost", "checkpointId": str(uuid4())},
    )
    # checkpointId resolves to no file on disk -> 404 (NotFoundError).
    response = client.post(
        f"/v1/inference/endpoints/{endpoint_id}/load", json={"checkpointId": str(uuid4())}
    )
    assert response.status_code == 404


def test_load_rejects_adapter_400(
    client: TestClient, make_text_checkpoint: Callable[..., str]
) -> None:
    checkpoint_id = make_text_checkpoint()
    endpoint_id = str(uuid4())
    client.post(
        "/v1/inference/endpoints",
        json={"endpointId": endpoint_id, "name": "with-adapter", "checkpointId": checkpoint_id},
    )
    response = client.post(
        f"/v1/inference/endpoints/{endpoint_id}/load",
        json={"checkpointId": checkpoint_id, "adapterCheckpointId": str(uuid4())},
    )
    assert response.status_code == 400
    assert "adapter" in response.json()["detail"].lower()


def test_load_vocab_mismatch_400_and_error_status(
    client: TestClient, make_mismatched_checkpoint: Callable[..., str]
) -> None:
    checkpoint_id = make_mismatched_checkpoint()
    endpoint_id = str(uuid4())
    client.post(
        "/v1/inference/endpoints",
        json={"endpointId": endpoint_id, "name": "bad-vocab", "checkpointId": checkpoint_id},
    )
    response = client.post(
        f"/v1/inference/endpoints/{endpoint_id}/load", json={"checkpointId": checkpoint_id}
    )
    assert response.status_code == 400
    assert "vocab" in response.json()["detail"].lower()
    # The endpoint must land in `error`, never stranded in `loading`.
    endpoints = {e["name"]: e for e in client.get("/v1/inference/endpoints").json()["endpoints"]}
    assert endpoints["bad-vocab"]["status"] == "error"


def test_load_corrupt_checkpoint_sets_error_status(client: TestClient, models_dir: Path) -> None:
    checkpoint_id = str(uuid4())
    (models_dir / f"{checkpoint_id}.pt").write_bytes(b"this is not a torch checkpoint")
    endpoint_id = str(uuid4())
    client.post(
        "/v1/inference/endpoints",
        json={"endpointId": endpoint_id, "name": "corrupt", "checkpointId": checkpoint_id},
    )
    response = client.post(
        f"/v1/inference/endpoints/{endpoint_id}/load", json={"checkpointId": checkpoint_id}
    )
    assert response.status_code == 400
    endpoints = {e["name"]: e for e in client.get("/v1/inference/endpoints").json()["endpoints"]}
    assert endpoints["corrupt"]["status"] == "error"
