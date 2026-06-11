"""The inference serving engine: endpoint lifecycle + OpenAI-compatible chat.

An *endpoint* is one served model — a name (the OpenAI ``model`` id clients
pass) bound to a checkpoint. The engine tracks endpoints, loads checkpoints
into eval-mode models, and answers chat/completions by rendering the messages
through the chat template, running the decode loop, and shaping the result
into the exact ``chat.completion`` / ``chat.completion.chunk`` wire format.

Concurrency: a re-entrant lock guards the endpoint registry. The multi-second
checkpoint load runs OUTSIDE the lock (transition to ``loading`` under lock,
load unlocked, commit under lock) so it never blocks listing or other
requests. Generation reads a stable reference to the loaded model and runs
unlocked — concurrent chats don't serialize.

The engine itself is torch-free to construct (it only holds dicts and reads
config). torch / tokenizers are imported lazily by the modules it calls, so
the control plane boots without them; a load or chat then fails with a clear,
specific error if they're missing.

Checkpoint resolution: a ``checkpointId`` resolves to a file under the
configured ``modelsDir`` — ``<modelsDir>/<checkpointId>.pt``, or
``<modelsDir>/<checkpointId>/latest.pt`` (or the newest ``*.pt`` in that
directory). Placing a trainer checkpoint there is how a trained model becomes
servable in v0.3; the coordinator's serve stage automates the copy later.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from .._generated.models import InferenceEndpoint, Status
from .chat_template import render_prompt_ids
from .checkpoint import CheckpointError, load_checkpoint
from .generate import generate, generate_stream
from .sampling import SamplingParams
from .tokenizer import InferenceTokenizer, TokenizerError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config import ConfigStore
    from .checkpoint import LoadedCheckpoint


class EngineError(Exception):
    """Base class for engine errors the routes map to HTTP status codes."""


class NotFoundError(EngineError):
    """A referenced endpoint / model / checkpoint does not exist (-> 404)."""


class BadRequestError(EngineError):
    """The request is well-formed but cannot be served (-> 400)."""


class ConflictError(EngineError):
    """The action conflicts with current state, e.g. a duplicate name (-> 409)."""


@dataclass
class _Slot:
    endpoint: InferenceEndpoint
    created: int
    loaded: LoadedCheckpoint | None = None
    tokenizer: InferenceTokenizer | None = None
    error: str | None = None


class InferenceEngine:
    def __init__(self, config: ConfigStore, *, device: str = "cpu") -> None:
        self._config = config
        self._device = device
        self._lock = threading.RLock()
        self._slots: dict[UUID, _Slot] = {}

    # ------------------------------------------------------------------ #
    # endpoint lifecycle
    # ------------------------------------------------------------------ #
    def create_endpoint(self, endpoint: InferenceEndpoint) -> InferenceEndpoint:
        with self._lock:
            if endpoint.endpointId in self._slots:
                raise ConflictError(f"endpoint {endpoint.endpointId} already exists")
            if any(s.endpoint.name == endpoint.name for s in self._slots.values()):
                raise ConflictError(f"endpoint name {endpoint.name!r} is already in use")
            ep = endpoint.model_copy(deep=True)
            ep.status = Status.unloaded
            self._slots[ep.endpointId] = _Slot(endpoint=ep, created=int(time.time()))
            return ep.model_copy(deep=True)

    def load_endpoint(
        self,
        endpoint_id: UUID,
        *,
        checkpoint_id: UUID,
        adapter_checkpoint_id: UUID | None = None,
    ) -> InferenceEndpoint:
        if adapter_checkpoint_id is not None:
            raise BadRequestError("adapter/LoRA checkpoints are not supported for serving in v0.3")

        with self._lock:
            slot = self._slots.get(endpoint_id)
            if slot is None:
                raise NotFoundError(f"endpoint {endpoint_id} not found")
            slot.endpoint.status = Status.loading
            slot.endpoint.checkpointId = checkpoint_id

        # Heavy work outside the lock. Any failure must (a) leave the endpoint in
        # `error`, never stranded in `loading`, and (b) surface as a typed engine
        # error the route can map — never a bare 500.
        try:
            path = self._resolve_checkpoint(checkpoint_id)  # NotFoundError -> 404
            loaded = load_checkpoint(path, map_location="cpu")
            loaded.model.to(self._device)
            tokenizer = (
                InferenceTokenizer.from_json(loaded.tokenizer_json)
                if loaded.tokenizer_json
                else None
            )
            if tokenizer is not None and tokenizer.vocab_size > loaded.architecture.vocabSize:
                raise CheckpointError(
                    f"tokenizer vocab ({tokenizer.vocab_size}) exceeds model vocab "
                    f"({loaded.architecture.vocabSize}); token ids would index past the "
                    "embedding. The checkpoint pairs a mismatched tokenizer and weights."
                )
        except EngineError:
            with self._lock:
                slot.endpoint.status = Status.error
                slot.error = "checkpoint not found"
            raise
        except (CheckpointError, TokenizerError) as e:
            with self._lock:
                slot.endpoint.status = Status.error
                slot.error = str(e)
            raise BadRequestError(str(e)) from e
        except Exception as e:  # never strand the endpoint in `loading`
            with self._lock:
                slot.endpoint.status = Status.error
                slot.error = str(e)
            raise BadRequestError(f"unexpected error loading checkpoint: {e}") from e

        with self._lock:
            slot.loaded = loaded
            slot.tokenizer = tokenizer
            slot.error = None
            slot.endpoint.status = Status.ready
            return slot.endpoint.model_copy(deep=True)

    def list_endpoints(self) -> list[InferenceEndpoint]:
        with self._lock:
            return [s.endpoint.model_copy(deep=True) for s in self._slots.values()]

    def get_endpoint(self, endpoint_id: UUID) -> InferenceEndpoint:
        with self._lock:
            slot = self._slots.get(endpoint_id)
            if slot is None:
                raise NotFoundError(f"endpoint {endpoint_id} not found")
            return slot.endpoint.model_copy(deep=True)

    def list_models(self) -> list[dict[str, Any]]:
        """OpenAI ``/v1/models`` ``data`` entries for every ready endpoint."""
        with self._lock:
            return [
                {
                    "id": s.endpoint.name,
                    "object": "model",
                    "created": s.created,
                    "owned_by": "eugene-plexus",
                }
                for s in self._slots.values()
                if s.endpoint.status == Status.ready
            ]

    # ------------------------------------------------------------------ #
    # serving
    # ------------------------------------------------------------------ #
    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[str]:
        with self._lock:
            slot = self._ready_slot_by_name(model)
            loaded = slot.loaded
            tokenizer = slot.tokenizer

        if loaded is None:  # pragma: no cover - _ready_slot_by_name guarantees loaded
            raise NotFoundError(f"model {model!r} is not loaded")
        if tokenizer is None:
            raise BadRequestError(
                f"model {model!r} has no tokenizer embedded in its checkpoint; "
                "cannot serve text chat"
            )

        try:
            import torch
        except ImportError as e:  # defensive: a ready endpoint already imported torch
            raise BadRequestError("serving requires PyTorch but it is not importable") from e

        params = self._resolve_params(temperature, top_p, max_tokens)
        prompt_ids = render_prompt_ids(
            messages, tokenizer, max_prompt_tokens=max(1, loaded.block_size - 1)
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)

        if stream:
            return self._stream(model, loaded, tokenizer, input_ids, params)

        result = generate(
            loaded.model,
            input_ids,
            params=params,
            eos_id=tokenizer.eos_id,
            block_size=loaded.block_size,
        )
        text = tokenizer.decode(result.token_ids)
        return _completion_response(
            model=model,
            text=text,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(result.token_ids),
            finish_reason=result.finish_reason,
        )

    def _stream(
        self,
        model: str,
        loaded: LoadedCheckpoint,
        tokenizer: InferenceTokenizer,
        input_ids: Any,
        params: SamplingParams,
    ) -> Iterator[str]:
        req_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())
        # Role-first chunk: delta carries the role and intentionally omits finish_reason.
        yield _sse_chunk(req_id, created, model, {"role": "assistant"}, include_finish=False)
        for piece, done, finish in generate_stream(
            loaded.model,
            input_ids,
            params=params,
            eos_id=tokenizer.eos_id,
            block_size=loaded.block_size,
            tokenizer=tokenizer,
        ):
            if done:
                yield _sse_chunk(req_id, created, model, {}, finish_reason=finish)
            elif piece:
                yield _sse_chunk(req_id, created, model, {"content": piece}, finish_reason=None)
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _ready_slot_by_name(self, name: str) -> _Slot:
        for slot in self._slots.values():
            if slot.endpoint.name == name:
                if slot.endpoint.status != Status.ready:
                    status = slot.endpoint.status
                    status_label = status.value if status is not None else "unknown"
                    raise BadRequestError(
                        f"model {name!r} is not ready (status: {status_label}); "
                        "load the endpoint first"
                    )
                return slot
        raise NotFoundError(f"model {name!r} not found")

    def _resolve_params(
        self, temperature: float | None, top_p: float | None, max_tokens: int | None
    ) -> SamplingParams:
        # `is not None` (not `or`) so a configured 0 — a legitimate value
        # (temperature 0 == greedy) — isn't silently replaced by the default.
        def _cfg(key: str, fallback: float) -> float:
            value = self._config.get(key)
            return float(value) if value is not None else fallback

        return SamplingParams(
            temperature=temperature if temperature is not None else _cfg("defaultTemperature", 0.7),
            top_p=top_p if top_p is not None else _cfg("defaultTopP", 1.0),
            top_k=0,
            repetition_penalty=1.0,
            max_tokens=max_tokens if max_tokens is not None else int(_cfg("defaultMaxTokens", 512)),
        )

    def _resolve_checkpoint(self, checkpoint_id: UUID) -> Path:
        models_dir = Path(self._config.get("modelsDir") or "inference-models")
        flat = models_dir / f"{checkpoint_id}.pt"
        if flat.exists():
            return flat
        folder = models_dir / str(checkpoint_id)
        if folder.is_dir():
            latest = folder / "latest.pt"
            if latest.exists():
                return latest
            pts = sorted(folder.glob("*.pt"))
            if pts:
                return pts[-1]
        raise NotFoundError(
            f"no checkpoint file for {checkpoint_id} under {models_dir} "
            f"(looked for {checkpoint_id}.pt and {checkpoint_id}/latest.pt)"
        )


def _completion_response(
    *, model: str, text: str, prompt_tokens: int, completion_tokens: int, finish_reason: str
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _sse_chunk(
    req_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    *,
    finish_reason: str | None = None,
    include_finish: bool = True,
) -> str:
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if include_finish:
        choice["finish_reason"] = finish_reason
    payload = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload)}\n\n"
