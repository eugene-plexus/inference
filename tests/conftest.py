"""Pytest fixtures + checkpoint factories for the inference test suite.

The factories produce *self-describing* checkpoints in the exact on-disk shape
the engine loads — ``{"model": state_dict, "meta": {"architecture": ...,
"tokenizer": {...}, "step": ...}}`` — so tests exercise the real load path
without depending on the trainer repo. ``arithmetic_checkpoint`` trains a tiny
model on a learnable next=prev+1 pattern so generation can be asserted to have
*learned* something (the end-to-end train->save->load->serve proof);
``text_checkpoint`` pairs a tiny (untrained) model with a real byte-level BPE
tokenizer so the OpenAI chat surface can be exercised end to end.

torch / tokenizers are imported inside the factories (test deps), keeping the
module import-light.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_inference.app import create_app
from eugene_plexus_inference.settings import Settings

_TINY_ARCH = {
    "modelType": "decoder_only",
    "nLayer": 2,
    "nHead": 2,
    "nKvHead": 1,
    "nEmbd": 32,
    "blockSize": 16,
}


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def settings(tmp_path: Path, models_dir: Path) -> Settings:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"modelsDir": str(models_dir)}))
    return Settings(config_file=config_path)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# checkpoint factories
# --------------------------------------------------------------------------- #


def _build_arch(vocab: int, **over: object) -> object:
    from eugene_plexus_inference._generated.common_models import ArchitectureConfig

    return ArchitectureConfig(**{**_TINY_ARCH, "vocabSize": vocab, **over})  # type: ignore[arg-type]


def _save_ckpt(
    path: Path, model: object, arch: object, *, tokenizer_json: str | None, step: int
) -> None:
    import torch

    payload = {
        "model": model.state_dict(),  # type: ignore[attr-defined]
        "training_state": {"iteration": step},
        "meta": {
            "architecture": arch.model_dump(mode="json"),  # type: ignore[attr-defined]
            "tokenizer": {
                "tokenizerId": None,
                "vocabFingerprint": None,
                "tokenizerJson": tokenizer_json,
            },
            "step": step,
            "valLoss": None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _train_arithmetic(model: object, *, vocab: int, seqlen: int, steps: int, seed: int = 0) -> None:
    """Teach the model next-token = (prev + 1) mod vocab — a pattern a tiny model
    learns in a few hundred steps, so greedy generation provably continues it."""
    import torch

    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)  # type: ignore[attr-defined]
    model.train()  # type: ignore[attr-defined]
    for _ in range(steps):
        starts = [rng.randrange(vocab) for _ in range(16)]
        rows = [[(s + i) % vocab for i in range(seqlen + 1)] for s in starts]
        batch = torch.tensor(rows, dtype=torch.long)
        _, loss = model(batch[:, :-1], batch[:, 1:])  # type: ignore[misc]
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()  # type: ignore[attr-defined]


def _make_tokenizer_json() -> tuple[str, int]:
    """Train a tiny byte-level BPE mirroring the data component's tokenizer."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tk = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tk.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tk.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=300,
        min_frequency=1,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "hello world, this is a small test corpus",
        "eugene plexus serves local language models",
        "training and inference share the same architecture",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    ]
    tk.train_from_iterator(corpus, trainer=trainer)
    return tk.to_str(), tk.get_vocab_size()


@pytest.fixture
def make_arithmetic_checkpoint(models_dir: Path) -> Callable[..., str]:
    """Factory -> checkpoint_id. Writes a trained, tokenizer-less checkpoint."""

    def _factory(*, vocab: int = 32, steps: int = 250) -> str:
        import torch

        from eugene_plexus_inference.engine.model import GPTModel

        torch.manual_seed(0)  # reproducible init -> deterministic learned argmax
        checkpoint_id = str(uuid4())
        arch = _build_arch(vocab)
        model = GPTModel(arch)  # type: ignore[arg-type]
        _train_arithmetic(model, vocab=vocab, seqlen=12, steps=steps)
        _save_ckpt(models_dir / f"{checkpoint_id}.pt", model, arch, tokenizer_json=None, step=steps)
        return checkpoint_id

    return _factory


@pytest.fixture
def make_text_checkpoint(models_dir: Path) -> Callable[..., str]:
    """Factory -> checkpoint_id. Writes an (untrained) model + real tokenizer so
    the full chat surface can run; semantic quality is not asserted."""

    def _factory() -> str:
        from eugene_plexus_inference.engine.model import GPTModel

        checkpoint_id = str(uuid4())
        tokenizer_json, vocab = _make_tokenizer_json()
        arch = _build_arch(vocab, blockSize=128)  # room for a chat prompt
        model = GPTModel(arch)  # type: ignore[arg-type]
        _save_ckpt(
            models_dir / f"{checkpoint_id}.pt", model, arch, tokenizer_json=tokenizer_json, step=0
        )
        return checkpoint_id

    return _factory


@pytest.fixture
def make_mismatched_checkpoint(models_dir: Path) -> Callable[..., str]:
    """Factory -> checkpoint_id whose embedded tokenizer vocab EXCEEDS the model
    vocab — an incoherent checkpoint the loader must reject at load time."""

    def _factory() -> str:
        from eugene_plexus_inference.engine.model import GPTModel

        checkpoint_id = str(uuid4())
        tokenizer_json, _big_vocab = _make_tokenizer_json()  # ~300 tokens
        arch = _build_arch(32)  # model vocab 32 << tokenizer vocab
        model = GPTModel(arch)  # type: ignore[arg-type]
        _save_ckpt(
            models_dir / f"{checkpoint_id}.pt", model, arch, tokenizer_json=tokenizer_json, step=0
        )
        return checkpoint_id

    return _factory


@pytest.fixture
def text_tokenizer() -> object:
    """A real byte-level BPE InferenceTokenizer for chat-template unit tests."""
    from eugene_plexus_inference.engine.tokenizer import InferenceTokenizer

    json_str, _ = _make_tokenizer_json()
    return InferenceTokenizer.from_json(json_str)


@pytest.fixture
def ready_text_endpoint(client: TestClient, make_text_checkpoint: Callable[..., str]) -> str:
    """Create + load a text endpoint named 'demo-model' via the HTTP surface,
    returning the model name clients pass as `model`."""
    checkpoint_id = make_text_checkpoint()
    endpoint_id = str(uuid4())
    created = client.post(
        "/v1/inference/endpoints",
        json={"endpointId": endpoint_id, "name": "demo-model", "checkpointId": checkpoint_id},
    )
    assert created.status_code == 201, created.text
    loaded = client.post(
        f"/v1/inference/endpoints/{endpoint_id}/load", json={"checkpointId": checkpoint_id}
    )
    assert loaded.status_code == 202, loaded.text
    assert loaded.json()["status"] == "ready"
    return "demo-model"
