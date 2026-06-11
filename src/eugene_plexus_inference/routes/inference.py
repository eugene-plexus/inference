"""Inference domain routes: OpenAI-compatible serving + endpoint management.

The handlers that touch the model are plain ``def`` (not ``async def``) so
FastAPI runs them in a worker thread — token generation is blocking CPU work
and must not stall the event loop. The streaming response wraps a synchronous
SSE generator, which FastAPI also drives from the threadpool.

When the engine is unavailable (safe mode, or a failed init that left the
component degraded) the read endpoints still answer with empty shapes — an
OpenAI client enumerating models gets a valid result — while the
mutating/serving endpoints return ``503`` with a ``Problem`` body.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from .._generated.common_models import Problem
from .._generated.models import (
    InferenceEndpoint,
    V1ChatCompletionsPostRequest,
    V1InferenceEndpointsEndpointIdLoadPostRequest,
    V1InferenceEndpointsGetResponse,
)
from ..engine.engine import (
    BadRequestError,
    ConflictError,
    EngineError,
    InferenceEngine,
    NotFoundError,
)

router = APIRouter(tags=["inference"])

_ERR_STATUS: list[tuple[type[EngineError], int]] = [
    (NotFoundError, status.HTTP_404_NOT_FOUND),
    (ConflictError, status.HTTP_409_CONFLICT),
    (BadRequestError, status.HTTP_400_BAD_REQUEST),
]


def _problem(status_code: int, title: str, detail: str) -> JSONResponse:
    slug = title.replace(" ", "-").lower()
    body = Problem(
        type=f"https://github.com/eugene-plexus/inference#{slug}",
        title=title,
        status=status_code,
        detail=detail,
        component="inference",
    )
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        content=body.model_dump(exclude_none=True),
    )


def _engine_error(e: EngineError) -> JSONResponse:
    code = next(
        (c for cls, c in _ERR_STATUS if isinstance(e, cls)), status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return _problem(code, type(e).__name__, str(e))


def _engine(request: Request) -> InferenceEngine | None:
    return getattr(request.app.state, "engine", None)


def _unavailable(request: Request) -> JSONResponse:
    detail = getattr(request.app.state, "engine_error", None) or "inference serving is unavailable"
    return _problem(status.HTTP_503_SERVICE_UNAVAILABLE, "Serving unavailable", detail)


# --------------------------------------------------------------------------- #
# OpenAI-compatible surface
# --------------------------------------------------------------------------- #


@router.post("/v1/chat/completions")
def create_chat_completion(request: Request, body: V1ChatCompletionsPostRequest) -> Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable(request)

    messages = [{"role": m.role.value, "content": m.content} for m in body.messages]
    stream = bool(body.stream)
    try:
        result = engine.chat_completion(
            model=body.model,
            messages=messages,
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
            stream=stream,
        )
    except EngineError as e:
        return _engine_error(e)

    if stream:
        return StreamingResponse(result, media_type="text/event-stream")
    return JSONResponse(content=result)


@router.get("/v1/models")
def list_models(request: Request) -> JSONResponse:
    """OpenAI-compatible model list — one entry per ready endpoint.

    Emits the full OpenAI Model shape (``id``, ``object``, ``created``,
    ``owned_by``) directly: the generated ``Datum`` only declares
    ``id``/``object``, but strict clients (e.g. openai-python) require
    ``created`` and ``owned_by``, so we bypass the narrow response_model.
    """
    engine = _engine(request)
    data = engine.list_models() if engine else []
    return JSONResponse(content={"object": "list", "data": data})


# --------------------------------------------------------------------------- #
# Endpoint management
# --------------------------------------------------------------------------- #


@router.get("/v1/inference/endpoints", response_model=V1InferenceEndpointsGetResponse)
def list_endpoints(request: Request) -> V1InferenceEndpointsGetResponse:
    engine = _engine(request)
    endpoints = engine.list_endpoints() if engine else []
    return V1InferenceEndpointsGetResponse(endpoints=endpoints)


@router.post("/v1/inference/endpoints", status_code=status.HTTP_201_CREATED)
def create_endpoint(request: Request, body: InferenceEndpoint) -> Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable(request)
    try:
        endpoint = engine.create_endpoint(body)
    except EngineError as e:
        return _engine_error(e)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=endpoint.model_dump(mode="json", exclude_none=True),
    )


@router.post("/v1/inference/endpoints/{endpoint_id}/load", status_code=status.HTTP_202_ACCEPTED)
def load_endpoint(
    request: Request,
    endpoint_id: UUID,
    body: V1InferenceEndpointsEndpointIdLoadPostRequest,
) -> Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable(request)
    try:
        endpoint = engine.load_endpoint(
            endpoint_id,
            checkpoint_id=body.checkpointId,
            adapter_checkpoint_id=body.adapterCheckpointId,
        )
    except EngineError as e:
        return _engine_error(e)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=endpoint.model_dump(mode="json", exclude_none=True),
    )
