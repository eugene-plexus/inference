"""Load a self-describing checkpoint into a ready-to-serve model.

A trainer checkpoint is a single ``.pt`` file holding at least::

    {
        "model": <state_dict>,
        "meta": {
            "architecture": <ArchitectureConfig as a JSON-able dict>,
            "tokenizer": {
                "tokenizerId": <str | None>,
                "vocabFingerprint": <str | None>,
                "tokenizerJson": <str | None>,   # full tokenizer.json text
            },
            ...
        },
    }

``meta.architecture`` is REQUIRED — it is how the inference engine rebuilds the
exact model graph standalone (no shared code or config with the trainer beyond
the schema). ``meta.tokenizer.tokenizerJson`` is required to serve *text*
chat (encode the prompt / decode the reply); without it the checkpoint can
only be loaded for raw-token use.

torch is imported lazily so this module can be imported by the control plane
without torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._generated.common_models import ArchitectureConfig

if TYPE_CHECKING:
    from torch import nn


class CheckpointError(Exception):
    """A checkpoint could not be loaded (missing, corrupt, or not self-describing)."""


@dataclass
class LoadedCheckpoint:
    """The product of loading a checkpoint: an eval-mode model plus the metadata
    needed to serve it (tokenizer text, architecture, step)."""

    model: nn.Module
    architecture: ArchitectureConfig
    tokenizer_json: str | None
    block_size: int
    eos_token: str
    step: int | None


def _strip_orig_mod(keys: set[str]) -> bool:
    return any(k.startswith("_orig_mod.") for k in keys)


def _reconcile(model: nn.Module, state: dict[str, Any]) -> dict[str, Any]:
    """Reconcile a checkpoint state-dict's keys with the model's, handling
    torch.compile's ``_orig_mod.`` prefix in either direction (a checkpoint
    saved from a compiled model loads into an uncompiled one and vice-versa)."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    model_compiled = _strip_orig_mod(model_keys)
    ckpt_compiled = _strip_orig_mod(ckpt_keys)
    if model_compiled and not ckpt_compiled:
        return {f"_orig_mod.{k}": v for k, v in state.items()}
    if ckpt_compiled and not model_compiled:
        return {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return state


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> LoadedCheckpoint:
    """Load ``path`` into an eval-mode model. Raises ``CheckpointError`` on any
    problem with a message specific enough for the operator to act on.

    ``weights_only=False`` is required because ``meta`` carries plain Python
    objects (the architecture dict, tokenizer text). Checkpoints are produced
    by the operator's own trainer and read from the operator's own host, so the
    trust assumption matches the rest of a personal install.
    """
    try:
        import torch

        from .model import GPTModel
    except ImportError as e:
        raise CheckpointError(f"serving requires PyTorch but it is not importable: {e}") from e

    if not path.exists():
        raise CheckpointError(f"checkpoint file not found: {path}")

    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:  # corrupt / unreadable / wrong-format file
        raise CheckpointError(f"failed to read checkpoint {path}: {e}") from e

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise CheckpointError(
            f"checkpoint {path} is missing the 'model' state-dict; not a trainer checkpoint"
        )

    meta = ckpt.get("meta") or {}
    arch_dict = meta.get("architecture")
    if not arch_dict:
        raise CheckpointError(
            f"checkpoint {path} is not self-describing: meta.architecture is absent. "
            "Re-train (or re-save) with a trainer that embeds the architecture into the "
            "checkpoint so it can be loaded for serving."
        )
    try:
        architecture = ArchitectureConfig.model_validate(arch_dict)
    except Exception as e:
        raise CheckpointError(f"checkpoint {path} has an invalid meta.architecture: {e}") from e

    try:
        model = GPTModel(architecture)
        model.load_state_dict(_reconcile(model, ckpt["model"]))
    except Exception as e:
        raise CheckpointError(
            f"checkpoint {path} weights do not match its declared architecture: {e}"
        ) from e
    model.eval()

    tok = meta.get("tokenizer") or {}
    tokenizer_json = tok.get("tokenizerJson")

    step_val = meta.get("step")
    step = int(step_val) if isinstance(step_val, int) else None

    return LoadedCheckpoint(
        model=model,
        architecture=architecture,
        tokenizer_json=tokenizer_json,
        block_size=architecture.blockSize,
        eos_token="</s>",
        step=step,
    )
