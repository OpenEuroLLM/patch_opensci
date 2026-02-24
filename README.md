# patch-opensci

Compatibility patches and inference tooling for the models of open-sci-ref-001, for instance
[open-sci/open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384](https://huggingface.co/open-sci/open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384)

This allows running those models on recent version of `transformers` e.g. 5.2+.

To convert a checkpoint for use with `transformers` 5.2+:

```bash
git clone https://github.com/OpenEuroLLM/patch_opensci && cd patch_opensci
uv run python download_model.py
uv run python hotfix_opensci.py --src_dir ./open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384
```

---

## What the hotfix does

The `open-sci` model was authored against a development snapshot of
`transformers` that introduced several APIs which break loading under both
older (4.x) and newer (5.x) released versions.

Running `hotfix_opensci.py` copies `<src_dir>` to `<src_dir>_fixed` and rewrites every `modeling_*.py`
inside it, producing a model that loads cleanly with transformers 5.2+ without
any runtime patching:

| Issue | Broken versions | Fix applied |
|---|---|---|
| `LossKwargs` removed | 5.0+ | Replace every occurrence of `LossKwargs` with `TransformersKwargs` |
| `ROPE_INIT_FUNCTIONS["default"]` removed | 5.0+ | Inline a fallback `_default_rope_init` function |
| `_tied_weights_keys` must be a `dict` | 5.0+ | Convert `["lm_head.weight"]` → `{"lm_head.weight": "model.embed_tokens.weight"}` |
| SDPA `enable_gqa=True` added for all MHA | 5.0+ | Inline a version-stable SDPA wrapper that never passes `enable_gqa` |
| Non-persistent `inv_freq` buffer zeroed on meta-device load | 5.0+ | Add a guard in `forward` that re-computes `inv_freq` from `rope_init_fn` if all-zero |


### `inference.py` — minimal 4.x shim

For 5.x, `inference.py` requires no patching when pointed at the `_fixed` model.

For 4.x, `inference.py` applies one small in-process shim before loading the model:

| Shim | Versions affected | Root cause |
|---|---|---|
| `TransformersKwargs` back-filled from `LossKwargs` | 4.48 – 4.57 | The model's remote code imports `TransformersKwargs`, which only exists in 5.0+ |

The `torch_dtype` → `dtype` rename is also handled transparently via a version
check, but this requires no patching of any library internals.

---

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/OpenEuroLLM/patch_opensci
cd patch_opensci
uv sync          # installs transformers==4.48.0 + torch + huggingface_hub
```

---

## Quickstart

### 1. Download the model

```bash
uv run python download_model.py
# saves to ./open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384
```

### 2. Run inference with transformers 4.x

Use `uv run --with transformers==<version>` to override the pinned version on the fly:

```bash
# run previous model on old transformer version
uv run --with transformers==4.48.0 python inference.py --model_path ./open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384
```

> The capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe

### 3. Produce a statically-patched copy for transformers 5.2+

```bash
uv run python hotfix_opensci.py --src_dir ./open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384
uv run --with transformers==5.2.0 python inference.py --model_path ./open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384_fixed
```

> The capital of France is Paris.

The hotfix-patched model produces bit-for-bit identical logits across all supported versions:

```
  ┌─────────┬──────────────┬──────────┬──────────────────┐
  │ Version │  Paris logit │ #2 logit │       gap        │
  ├─────────┼──────────────┼──────────┼──────────────────┤
  │ 4.48.0  │ 13.0544      │ 10.1476  │ 2.91             │
  ├─────────┼──────────────┼──────────┼──────────────────┤
  │ 4.57.6  │ 13.0544      │ 10.1476  │ 2.91 (identical) │
  ├─────────┼──────────────┼──────────┼──────────────────┤
  │ 5.2.0   │ 13.0544      │ 10.1476  │ 2.91 (identical) │
  └─────────┴──────────────┴──────────┴──────────────────┘
```

All three versions are bit-for-bit identical after the hotfix.

---

## Evaluation

We evaluate the model fix:

```
                                                       arc_challenge  arc_easy     boolq  commonsense_qa  copa  hellaswag  lambada_openai      mmlu  openbookqa      piqa  social_iqa  winogrande
open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384_fixed_new       0.512799  0.800505  0.793578        0.624079  0.83   0.724358        0.598292  0.508125        0.43  0.789445    0.444217    0.636938                                     
open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384_original        0.485495  0.787458  0.762080        0.314496  0.80   0.687313        0.566854  0.396169        0.39  0.772579         NaN    0.621152                                     
```

where `_original` is the results reported in opensci-ref models obtained with transformer version bellow 4.48 and
`_fixed_new` is used to indicate the model fixed with this patch evaluated with 5.2.0 transformers version.  


---

## Running the tests

```bash
uv run pytest
```

Tests spawn a subprocess per transformers version and assert that the
completion contains `"Paris"`:

```
test_inference_4x[4.48.0]   PASSED   transformers 4.48.0  (original model + TransformersKwargs shim)
test_inference_4x[4.49.0]   PASSED   transformers 4.49.0  (original model + TransformersKwargs shim)
test_inference_4x[4.57.6]   PASSED   transformers 4.57.6  (original model + TransformersKwargs shim)
test_inference_5x[5.0.0]    XFAIL    known upstream regression in 5.0/5.1
test_inference_5x[5.2.0]    PASSED   transformers 5.2.0   (hotfix-patched model, no runtime patching)
```


---

## Transformers version support

| Version range | Status | Notes |
|---|---|---|
| 4.48.x | ✅ Supported | Pinned project default |
| 4.49 – 4.57 | ✅ Supported | `LossKwargs` present, `TransformersKwargs` absent |
| 5.0 – 5.1 | ❌ Not supported | Known upstream `generate()` regression; unrelated to this patch |
| 5.2+ | ✅ Supported | Use the hotfix-patched model (see below) |

---