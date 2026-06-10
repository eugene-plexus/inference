"""Auth state for the inference component's verify-only role.

Built once at startup from the env vars the watchdog threads in when
it spawns this child:

  * `EUGENE_PLEXUS_INF_AUTH_SIGNING_KEY` — base64 of the 32-byte HMAC
    key used to validate inbound bearer tokens.
  * `EUGENE_PLEXUS_INF_SERVICE_TOKEN` — long-lived JWT (`aud:
    service:inference`). The watchdog supplies this for symmetry with
    other kinds; a future serving engine can present it for outbound
    peer calls (e.g. fetching a checkpoint from the trainer to load).
  * `EUGENE_PLEXUS_INF_MASTER_KEY` — base64 of the 32-byte secretbox
    key. Not used in the v0.3 skeleton — no inference secrets are
    encrypted at rest yet. Reserved.

If `AUTH_SIGNING_KEY` is unset, the component runs in
`auth_disabled=True` mode (dev / standalone path). Production via the
watchdog always supplies the env var.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthState:
    signing_key: bytes | None
    service_token: str | None
    master_key: bytes | None

    @property
    def auth_disabled(self) -> bool:
        return self.signing_key is None


def _decode_b64_key(value: str | None, *, expected_len: int, label: str) -> bytes | None:
    if not value:
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as e:
        raise ValueError(f"{label}: not valid base64 ({e})") from e
    if len(raw) != expected_len:
        raise ValueError(
            f"{label}: expected {expected_len} bytes after base64-decode, got {len(raw)}"
        )
    return raw


def load_auth_state(
    *,
    signing_key_b64: str | None,
    service_token: str | None,
    master_key_b64: str | None,
) -> AuthState:
    signing_key = _decode_b64_key(signing_key_b64, expected_len=32, label="AUTH_SIGNING_KEY")
    master_key = _decode_b64_key(master_key_b64, expected_len=32, label="MASTER_KEY")

    if signing_key is None:
        if service_token or master_key:
            raise ValueError(
                "auth env vars inconsistent: SERVICE_TOKEN or MASTER_KEY is set but "
                "AUTH_SIGNING_KEY is not — refusing to start in a partially-auth state"
            )
        log.warning(
            "EUGENE_PLEXUS_INF_AUTH_SIGNING_KEY not set — running unauthenticated "
            "(dev/standalone mode). Production spawns via watchdog always supply this."
        )
        return AuthState(signing_key=None, service_token=None, master_key=None)

    return AuthState(
        signing_key=signing_key,
        service_token=service_token or None,
        master_key=master_key,
    )
