"""Engine-level tests: checkpoint loading + the decode loop.

The headline test trains a tiny model on a learnable pattern, saves a
self-describing checkpoint, loads it through the real loader, and asserts
greedy generation *continues the learned pattern* — the train -> save -> load
-> serve proof that the served weights are the trained ones. Stop conditions
are pinned with deterministic stub models.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

from eugene_plexus_inference.engine.checkpoint import CheckpointError, load_checkpoint
from eugene_plexus_inference.engine.generate import generate
from eugene_plexus_inference.engine.sampling import SamplingParams


class _StubLM(nn.Module):
    """Returns logits that always favor one token id — to pin stop behavior."""

    def __init__(self, vocab: int, favored: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.favored = favored

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        b, t = idx.shape
        logits = torch.zeros(b, t, self.vocab)
        logits[..., self.favored] = 50.0
        return logits, None


def _greedy(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_tokens=max_tokens
    )


def test_load_checkpoint_rebuilds_model(
    make_arithmetic_checkpoint: Callable[..., str], models_dir: Path
) -> None:
    checkpoint_id = make_arithmetic_checkpoint(vocab=32)
    loaded = load_checkpoint(models_dir / f"{checkpoint_id}.pt")
    assert loaded.architecture.vocabSize == 32
    assert loaded.block_size == 16
    assert loaded.eos_token == "</s>"


def test_generate_continues_learned_pattern(
    make_arithmetic_checkpoint: Callable[..., str], models_dir: Path
) -> None:
    checkpoint_id = make_arithmetic_checkpoint(vocab=32, steps=300)
    loaded = load_checkpoint(models_dir / f"{checkpoint_id}.pt")
    prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    result = generate(
        loaded.model, prompt, params=_greedy(5), eos_id=9999, block_size=loaded.block_size
    )
    # The model learned next = prev + 1 (mod 32): 8 -> 9, 10, 11, 12, 13.
    assert result.token_ids == [9, 10, 11, 12, 13]


def test_generate_stops_on_eos() -> None:
    model = _StubLM(vocab=16, favored=3)
    result = generate(model, torch.tensor([[0, 1]]), params=_greedy(8), eos_id=3, block_size=8)
    # EOS terminates without being recorded — it is neither a content token nor
    # a usage token, and this matches generate_stream exactly.
    assert result.token_ids == []
    assert result.finish_reason == "stop"


def test_generate_hits_length_budget() -> None:
    model = _StubLM(vocab=16, favored=0)
    result = generate(model, torch.tensor([[0, 1]]), params=_greedy(4), eos_id=3, block_size=8)
    assert result.token_ids == [0, 0, 0, 0]
    assert result.finish_reason == "length"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_non_self_describing_raises(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"model": {}, "meta": {}}, path)  # no meta.architecture
    with pytest.raises(CheckpointError, match="self-describing"):
        load_checkpoint(path)


def test_load_arch_weight_mismatch_raises(
    make_arithmetic_checkpoint: Callable[..., str], models_dir: Path
) -> None:
    checkpoint_id = make_arithmetic_checkpoint(vocab=32)
    path = models_dir / f"{checkpoint_id}.pt"
    ckpt = torch.load(path, weights_only=False)
    # Corrupt the embedded architecture so the weights no longer fit.
    ckpt["meta"]["architecture"]["nEmbd"] = 64
    torch.save(ckpt, path)
    with pytest.raises(CheckpointError, match="do not match"):
        load_checkpoint(path)


def test_resolve_params_honors_zero_temperature() -> None:
    """A configured `defaultTemperature: 0` (greedy) must not be replaced by the
    default via an or-falsy bug."""
    from eugene_plexus_inference.engine.engine import InferenceEngine

    class _Cfg:
        def get(self, key: str) -> object:
            return {"defaultTemperature": 0, "defaultTopP": 0, "defaultMaxTokens": 7}.get(key)

    engine = InferenceEngine(_Cfg(), device="cpu")  # type: ignore[arg-type]
    params = engine._resolve_params(None, None, None)
    assert params.temperature == 0.0
    assert params.top_p == 0.0
    assert params.max_tokens == 7
