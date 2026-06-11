"""Thin wrapper over a Hugging Face ``tokenizers`` Tokenizer loaded from text.

The data component trains a byte-level BPE tokenizer and persists it as a
``tokenizer.json``; the trainer embeds that text into each checkpoint's
``meta.tokenizer.tokenizerJson`` so the inference engine can encode prompts and
decode replies without reaching back into the data component. Special tokens
are ``<pad> <unk> <s> </s>`` with EOS = ``</s>``.

``tokenizers`` is imported lazily so this module imports cleanly without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer

EOS_TOKEN = "</s>"
BOS_TOKEN = "<s>"


class TokenizerError(Exception):
    """The tokenizer could not be loaded or a token was missing."""


class InferenceTokenizer:
    """Encode text -> token ids and decode ids -> text, with EOS handling."""

    def __init__(self, tk: Tokenizer) -> None:
        self._tk = tk
        eos_id = tk.token_to_id(EOS_TOKEN)
        if eos_id is None:
            raise TokenizerError(f"tokenizer has no {EOS_TOKEN!r} token; cannot detect end-of-text")
        self.eos_id: int = eos_id
        self.bos_id: int | None = tk.token_to_id(BOS_TOKEN)

    @classmethod
    def from_json(cls, json_str: str) -> InferenceTokenizer:
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as e:  # pragma: no cover - dep-present in CI/runtime
            raise TokenizerError(
                "the 'tokenizers' package is required to serve text but is not installed"
            ) from e
        try:
            tk = Tokenizer.from_str(json_str)
        except Exception as e:
            raise TokenizerError(f"could not parse the embedded tokenizer.json: {e}") from e
        return cls(tk)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode ``text`` to ids. The chat template controls framing itself, so
        callers pass ``add_special_tokens=False`` to suppress the tokenizer's
        own BOS/EOS post-processor and avoid double-framing."""
        return self._tk.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self._tk.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()
