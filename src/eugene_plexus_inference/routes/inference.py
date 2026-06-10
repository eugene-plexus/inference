"""Inference domain routes: OpenAI-compatible serving + endpoint management.

v0.3 SKELETON. The real serving engine is not implemented yet, so no
model is loaded. The endpoints that need a running engine return `501
Not Implemented` with a standard `Problem` body:

  * `POST /v1/chat/completions` — needs a loaded model to generate.
  * `POST /v1/inference/endpoints` — needs the engine to register a
    served checkpoint.
  * `POST /v1/inference/endpoints/{endpointId}/load` — needs the engine
    to load weights.

The endpoints whose answer doesn't depend on a running engine are real:

  * `GET /v1/models` returns the OpenAI-compatible empty list shape
    (`{object: list, data: []}`) — nothing loaded yet.
  * `GET /v1/inference/endpoints` returns an empty endpoint list.

When the engine lands it replaces the 501s; the wire shapes here are the
long-term contract.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response, status

from .._generated.common_models import Problem
from .._generated.models import (
    Datum,
    InferenceEndpoint,
    V1ChatCompletionsPostRequest,
    V1InferenceEndpointsEndpointIdLoadPostRequest,
    V1InferenceEndpointsGetResponse,
    V1ModelsGetResponse,
)

router = APIRouter(tags=["inference"])

_ENGINE_NOT_IMPLEMENTED = (
    "inference serving engine not implemented in the v0.3 skeleton; "
    "this repo ships the control-plane wire shape only and no model is loaded"
)


def _not_implemented(operation: str) -> Response:
    problem = Problem(
        type="https://github.com/eugene-plexus/inference#engine-not-implemented",
        title="Inference engine not implemented",
        status=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"{operation}: {_ENGINE_NOT_IMPLEMENTED}.",
        component="inference",
    )
    return Response(
        content=problem.model_dump_json(exclude_none=True),
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        media_type="application/problem+json",
    )


# --------------------------------------------------------------------------- #
# OpenAI-compatible surface
# --------------------------------------------------------------------------- #


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    body: V1ChatCompletionsPostRequest,
) -> Response:
    return _not_implemented("createChatCompletion")


@router.get("/v1/models", response_model=V1ModelsGetResponse)
async def list_models(request: Request) -> V1ModelsGetResponse:
    """OpenAI-compatible model list. The skeleton has no serving engine
    and therefore no loaded models — returns the empty-list shape rather
    than 501 so OpenAI clients enumerating models get a valid result."""
    data: list[Datum] = []
    return V1ModelsGetResponse(object="list", data=data)


# --------------------------------------------------------------------------- #
# Endpoint management
# --------------------------------------------------------------------------- #


@router.get("/v1/inference/endpoints", response_model=V1InferenceEndpointsGetResponse)
async def list_endpoints(request: Request) -> V1InferenceEndpointsGetResponse:
    """List inference endpoints. The skeleton has no engine and therefore
    no endpoints — returns an empty list rather than 501 so callers
    polling for endpoint state get a valid empty result."""
    return V1InferenceEndpointsGetResponse(endpoints=[])


@router.post("/v1/inference/endpoints", status_code=status.HTTP_201_CREATED)
async def create_endpoint(request: Request, body: InferenceEndpoint) -> Response:
    return _not_implemented("createEndpoint")


@router.post("/v1/inference/endpoints/{endpoint_id}/load", status_code=202)
async def load_endpoint(
    request: Request,
    endpoint_id: UUID,
    body: V1InferenceEndpointsEndpointIdLoadPostRequest,
) -> Response:
    return _not_implemented("loadEndpoint")
