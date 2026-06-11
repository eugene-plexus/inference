# eugene-plexus-inference

Local model serving with OpenAI-compatible endpoints for [Eugene Plexus](https://github.com/eugene-plexus).

## What this is

The inference component of Eugene Plexus. It serves trained checkpoints
locally behind an **OpenAI-compatible** API so existing tooling can talk to
a model the platform produced without changes. It loads a checkpoint (and an
optional LoRA adapter) into an endpoint, then answers `/v1/chat/completions`
(streaming + non-streaming) and reports loaded models through `/v1/models`. It
references checkpoints produced by the `trainer` component via
`common.yaml#/components/schemas/Checkpoint`.

```
POST   /v1/chat/completions                       OpenAI-compatible chat completion
GET    /v1/models                                 OpenAI-compatible model list
GET    /v1/inference/endpoints                     list inference endpoints
POST   /v1/inference/endpoints                      create an endpoint backed by a checkpoint
POST   /v1/inference/endpoints/{endpointId}/load    load / hot-swap the served checkpoint/adapter
```

Plus the standard Eugene Plexus config trio (`GET /v1/config`,
`GET /v1/config/schema`, `PATCH /v1/config`), `POST /v1/config/test`,
`POST /v1/admin/restart`, and `GET /healthz`.

## Serving engine

The serving engine is implemented. It loads a **self-describing checkpoint**
(the trainer embeds the model's `ArchitectureConfig` and tokenizer into the
checkpoint, so the inference host rebuilds and serves it standalone),
renders chat messages through a ChatML-style template, and decodes with
temperature / top-p / top-k sampling (greedy at `temperature=0`). Endpoints
move through `unloaded → loading → ready`; a ready endpoint's `name` is the
OpenAI `model` id clients pass.

Make a checkpoint servable by placing it under the configured `modelsDir` as
`<checkpointId>.pt` (or `<checkpointId>/latest.pt`), then create an endpoint
and load it. The coordinator's serve stage automates this copy later.

**v0.3 first-cut limits** (each a clean follow-up): CPU decode only (no GPU
device selection yet); **no KV cache** — full-recompute decode, fine for the
small local models this targets but `O(n²)` in generated length; single
process; LoRA/adapter serving is rejected (`400`). A base (pretrained)
checkpoint does text *continuation* — it only follows the chat template once
it has been fine-tuned on one.

## Quick start

```bash
pip install -e ".[dev]"
python -m eugene_plexus_inference
# default port 8090; override via PATCH /v1/config or the config file
```

The first run creates a `config.yaml` in the working directory with the
component's defaults. Edit through the UI, through `PATCH /v1/config`, or
by hand.

## Degraded-mode startup

Per the project-wide rule (`feedback_degraded_mode_required.md`), a bad
config never prevents the component from starting. Config endpoints stay
reachable so operators can fix the broken setting through the UI. The
engine builds without torch present (it's imported lazily on first
load/chat), so the control plane always comes up; in **safe mode**
(`EUGENE_PLEXUS_INF_SAFE_MODE=1`) serving is disabled and the
serving/endpoint-mutating routes return `503` while config stays editable.

## Codegen

Pydantic models for the inference component and shared schemas are
generated from the pinned `eugene-plexus/specs` commit:

```bash
python scripts/codegen.py
```

`SPECS_REF` records the commit SHA. Bump it to track a newer specs
release; CI re-runs codegen and fails if the working tree drifts.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and
[`CONTRIBUTING.md`](CONTRIBUTING.md) (DCO sign-off required).
