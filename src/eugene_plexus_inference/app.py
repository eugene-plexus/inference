"""FastAPI app factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from . import __version__
from .auth_state import load_auth_state
from .config import ConfigStore
from .dependencies import require_authorized, require_operator
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import health as health_routes
from .routes import inference as inference_routes
from .settings import Settings, load_settings

log = logging.getLogger(__name__)

_SAFE_MODE_ENGINE_ERROR = (
    "safe mode active (EUGENE_PLEXUS_INF_SAFE_MODE=1): model serving is disabled; "
    "config endpoints remain reachable. Fix config and restart without the env var."
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    config_store = ConfigStore(settings.config_file)
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_INF_SAFE_MODE=1); ignoring "
            "%s and running on defaults. Fix config via /v1/config, then "
            "restart without the env var.",
            settings.config_file,
        )
    else:
        config_store.load()
    app.state.config_store = config_store
    app.state.safe_mode = settings.safe_mode

    # v0.2 auth state. Tests can pre-populate `app.state.auth_state` to
    # exercise authed paths; the default lifespan build reads env vars
    # via Settings and produces an auth-disabled state when the watchdog
    # didn't supply AUTH_SIGNING_KEY.
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            signing_key_b64=settings.auth_signing_key,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )

    # Build the serving engine. Construction is torch-free (it only holds the
    # endpoint registry and reads config); torch / tokenizers are imported
    # lazily on the first load/chat, so the control plane boots even without
    # them. In safe mode we leave the engine unbuilt so serving is disabled
    # but config stays reachable; any unexpected build failure also degrades
    # rather than crashing, per feedback_degraded_mode_required.md. Tests may
    # pre-populate `app.state.engine`.
    if not hasattr(app.state, "engine"):
        if settings.safe_mode:
            app.state.engine = None
            app.state.engine_error = _SAFE_MODE_ENGINE_ERROR
        else:
            try:
                from .engine.engine import InferenceEngine

                app.state.engine = InferenceEngine(config_store, device="cpu")
                app.state.engine_error = None
            except Exception as e:  # pragma: no cover - defensive degrade
                log.exception("failed to build the inference engine; starting degraded")
                app.state.engine = None
                app.state.engine_error = f"engine initialization failed: {e}"

    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — inference",
        description=(
            "Local model serving with OpenAI-compatible chat/completions "
            "endpoints. Loads checkpoints produced by the trainer and serves "
            "them; CPU decode with no KV cache in this first cut."
        ),
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Health stays unauthenticated — supervisors and load balancers need
    # to probe it without holding credentials.
    app.include_router(health_routes.router)

    # Config edits are operator-only — service tokens are rejected so a
    # compromised peer can't reconfigure the inference component (e.g.
    # repoint the models directory).
    operator = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator)
    app.include_router(admin_routes.router, dependencies=operator)

    # Serving + endpoint management: the orchestrator (service) drives
    # inference; operators may also manage endpoints through the UI.
    app.include_router(inference_routes.router, dependencies=[Depends(require_authorized)])

    return app
