"""Unit tests for the logit-to-token sampler."""

from __future__ import annotations

import torch

from eugene_plexus_inference.engine.sampling import SamplingParams, sample_next_token


def _params(**over: object) -> SamplingParams:
    base = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_tokens": 8,
    }
    return SamplingParams(**{**base, **over})  # type: ignore[arg-type]


def test_greedy_picks_argmax() -> None:
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])
    assert sample_next_token(logits, _params(temperature=0.0), generated_ids=[]) == 1


def test_top_k_one_is_deterministic_argmax() -> None:
    logits = torch.tensor([0.1, 0.2, 9.0, 0.3])
    # top_k=1 leaves only the max in the nucleus, so any sampling picks it.
    assert sample_next_token(logits, _params(top_k=1), generated_ids=[]) == 2


def test_top_p_tiny_collapses_to_argmax() -> None:
    logits = torch.tensor([10.0, 1.0, 0.5, 0.2])
    # A near-zero top_p keeps only the single highest-prob token.
    assert sample_next_token(logits, _params(top_p=0.01), generated_ids=[]) == 0


def test_repetition_penalty_downweights_repeats() -> None:
    # Token 0 has the highest raw logit, but it was just generated; a strong
    # penalty divides its positive logit below token 1, which then wins greedily.
    logits = torch.tensor([10.0, 9.0, 0.0])
    chosen = sample_next_token(
        logits, _params(temperature=0.0, repetition_penalty=2.0), generated_ids=[0]
    )
    assert chosen == 1
