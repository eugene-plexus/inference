"""Tests for v0.2 bearer auth on the inference component.

Same verify-only pattern as orchestrator and hemisphere-driver. Tests
mint JWTs directly with PyJWT against a known signing key (standing in
for the watchdog) and assert the dependency layer accepts/rejects the
right shapes.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from pathlib import Path

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_inference.app import create_app
from eugene_plexus_inference.auth_state import AuthState
from eugene_plexus_inference.settings import Settings

_JWT_ALG = "HS256"


def _issue(
    *,
    signing_key: bytes,
    sub: str,
    aud: str,
    ttl_seconds: int = 60,
    iat: int | None = None,
) -> str:
    issued_at = iat if iat is not None else int(time.time())
    claims = {
        "sub": sub,
        "aud": aud,
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
    }
    return jwt.encode(claims, signing_key, algorithm=_JWT_ALG)


@pytest.fixture
def signing_key() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def authed_app(tmp_path: Path, signing_key: bytes) -> FastAPI:
    settings = Settings(config_file=tmp_path / "config.yaml")
    app = create_app(settings=settings)
    app.state.auth_state = AuthState(
        signing_key=signing_key,
        service_token=_issue(
            signing_key=signing_key,
            sub="inference",
            aud="service:inference",
            ttl_seconds=365 * 24 * 3600,
        ),
        master_key=None,
    )
    return app


@pytest.fixture
def authed_client(authed_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(authed_app) as c:
        yield c


@pytest.fixture
def operator_token(signing_key: bytes) -> str:
    return _issue(signing_key=signing_key, sub="operator", aud="operator")


@pytest.fixture
def orchestrator_service_token(signing_key: bytes) -> str:
    return _issue(signing_key=signing_key, sub="orchestrator", aud="service:orchestrator")


# --------------------------------------------------------------------------- #
# Auth-disabled (default fixtures) — unchanged behavior
# --------------------------------------------------------------------------- #


def test_auth_disabled_lets_everything_through(client: TestClient) -> None:
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/config").status_code == 200


# --------------------------------------------------------------------------- #
# Health always open
# --------------------------------------------------------------------------- #


def test_healthz_is_always_open(authed_client: TestClient) -> None:
    assert authed_client.get("/healthz").status_code == 200


# --------------------------------------------------------------------------- #
# Missing / wrong-key / expired tokens reject with Problem JSON
# --------------------------------------------------------------------------- #


def test_missing_bearer_rejects_with_401(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config")
    assert response.status_code == 401
    assert response.json()["detail"]["component"] == "inference"


def test_wrong_signing_key_rejects(authed_client: TestClient) -> None:
    other = secrets.token_bytes(32)
    token = _issue(signing_key=other, sub="operator", aud="operator")
    response = authed_client.get("/v1/config", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_expired_token_rejects(authed_client: TestClient, signing_key: bytes) -> None:
    expired = _issue(
        signing_key=signing_key,
        sub="operator",
        aud="operator",
        ttl_seconds=-60,
        iat=int(time.time()) - 120,
    )
    response = authed_client.get("/v1/config", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401


# --------------------------------------------------------------------------- #
# Operator audience accepted on operator routes + serving routes
# --------------------------------------------------------------------------- #


def test_operator_token_accepted_on_config(authed_client: TestClient, operator_token: str) -> None:
    response = authed_client.get(
        "/v1/config", headers={"Authorization": f"Bearer {operator_token}"}
    )
    assert response.status_code == 200


def test_operator_token_accepted_on_endpoints(
    authed_client: TestClient, operator_token: str
) -> None:
    response = authed_client.get(
        "/v1/inference/endpoints", headers={"Authorization": f"Bearer {operator_token}"}
    )
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Service audience rejected on config, accepted on the serving surface
# --------------------------------------------------------------------------- #


def test_service_token_rejected_on_config_patch(
    authed_client: TestClient, orchestrator_service_token: str
) -> None:
    """A compromised peer must not be able to rewrite the inference
    component's config (e.g. repoint the models directory)."""
    response = authed_client.patch(
        "/v1/config",
        json={"defaultMaxTokens": 1},
        headers={"Authorization": f"Bearer {orchestrator_service_token}"},
    )
    assert response.status_code == 401


def test_service_token_accepted_on_endpoints(
    authed_client: TestClient, orchestrator_service_token: str
) -> None:
    """Normal production path: the orchestrator lists endpoints / serves
    inference using its service:orchestrator token."""
    response = authed_client.get(
        "/v1/inference/endpoints",
        headers={"Authorization": f"Bearer {orchestrator_service_token}"},
    )
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# load_auth_state contract
# --------------------------------------------------------------------------- #


def test_load_auth_state_disabled_when_no_signing_key() -> None:
    from eugene_plexus_inference.auth_state import load_auth_state

    state = load_auth_state(signing_key_b64=None, service_token=None, master_key_b64=None)
    assert state.auth_disabled is True


def test_load_auth_state_rejects_partial_auth() -> None:
    from eugene_plexus_inference.auth_state import load_auth_state

    with pytest.raises(ValueError, match="inconsistent"):
        load_auth_state(
            signing_key_b64=None,
            service_token="dummy",
            master_key_b64=None,
        )


def test_load_auth_state_rejects_wrong_length_signing_key() -> None:
    import base64

    from eugene_plexus_inference.auth_state import load_auth_state

    short = base64.b64encode(b"\x00" * 16).decode("ascii")
    with pytest.raises(ValueError, match="32 bytes"):
        load_auth_state(
            signing_key_b64=short,
            service_token=None,
            master_key_b64=None,
        )
