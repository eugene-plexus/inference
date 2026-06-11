"""The inference serving engine.

Loads checkpoints produced by the trainer component and serves them behind an
OpenAI-compatible chat/completions surface. torch and tokenizers are imported
lazily (inside the modules that need them) so the control plane — config,
health, endpoint listing — boots even when those heavyweight dependencies are
absent; a load/generate request then fails with a clear error instead of the
whole service refusing to start.

Module layout mirrors the trainer's ``engine`` package for the shared model
code (``model``/``attention``/``layers``/``block`` are copied verbatim so a
checkpoint's state-dict keys line up exactly), plus the serving-specific
modules: ``checkpoint`` (load a self-describing checkpoint), ``tokenizer``
(encode/decode), ``sampling`` (logits -> next token), ``chat_template``
(messages -> prompt), ``generate`` (autoregressive decode loop), ``registry``
(loaded-endpoint lifecycle), and ``engine`` (the top-level facade the routes
drive).
"""

from __future__ import annotations
