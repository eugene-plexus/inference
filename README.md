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

## v0.3 skeleton status

This repo currently ships the **control-plane skeleton**: the HTTP wire
shape (routes + generated models + config + auth + health + safe mode) is
complete, but the actual serving engine is **not implemented yet** and no
model is loaded. The engine-dependent endpoints
(`POST /v1/chat/completions`, `POST /v1/inference/endpoints`,
`POST /v1/inference/endpoints/{endpointId}/load`) return `501 Not
Implemented` with a standard `Problem` body explaining that the serving
engine is future work. `GET /v1/models` returns the OpenAI-compatible
empty list (`{object: list, data: []}`) and `GET /v1/inference/endpoints`
returns an empty endpoint list.

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
reachable so operators can fix the broken setting through the UI;
serving endpoints behave according to the skeleton (501) until the
serving engine lands.

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
