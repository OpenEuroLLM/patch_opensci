# Notes: Fixing open-sci-ref logit differences across transformers versions

## Background

The `open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384` model was authored against a
development snapshot of `transformers` and fails to load under both older (4.x) and
newer (5.x) released versions.  Four structural issues were addressed first
(`LossKwargs` → `TransformersKwargs`, inlining the ROPE `"default"` fallback,
converting `_tied_weights_keys` from list to dict, and a `TransformersKwargs`
back-fill shim in the inference scripts for 4.x).  After those fixes the model loaded
and generated text under both families, but a fifth, more subtle issue caused the
logits to differ by ~0.66 between 4.x and 5.x — meaning different tokens were
predicted and greedy decoding diverged after just one token.

---

## Approaches tried

### 1. `apply_rotary_pos_emb` — ruled out

The first hypothesis was that the rotary position embedding helper had changed
between library versions.  Inspection of the model file confirmed that
`apply_rotary_pos_emb` is defined entirely within the custom `modeling_opensci.py`
and imports nothing from `transformers`; it is identical in every version.

### 2. ROPE `inv_freq` comparison via hooks — misleading result

A hook was attached to `OpensciRotaryEmbedding.forward` and the `inv_freq` buffer was
printed in both versions.  The hook appeared to show identical values (`[1.0, 0.649,
0.422, ...]`) because the hook was reading the buffer *before* the meta-device
materialization had time to zero it.  This led to the incorrect conclusion that the
frequencies were fine.

### 3. SDPA `enable_gqa` — partially correct, not sufficient

Inspection of `transformers` 5.x's SDPA dispatch wrapper revealed that it now
unconditionally passes `enable_gqa=True` to `torch.nn.functional.scaled_dot_product_attention`,
even for plain multi-head attention (where `num_key_value_groups = 1`).  This routes
to a different PyTorch kernel than 4.x and can shift logits.  A patch was written
(Patch #4) that inlines a version-stable SDPA wrapper omitting `enable_gqa`.  The
patch was applied and confirmed present, but re-running the logit check still showed
5.x at 12.3956 versus 4.x at 13.0544 — so the SDPA difference was a real but
secondary issue; something else was still broken.

### 4. Forced eager attention — ruled out attention backend

To isolate attention as the cause, the model was loaded under 5.x with
`attn_implementation="eager"`, bypassing SDPA entirely.  The logit remained at 12.3956,
confirming the divergence was upstream of the attention computation.

### 5. Weight tensor comparison — all 435 tensors identical

All 435 named parameters and buffers were serialised with `torch.save` under 4.x and
5.x and compared element-wise.  Every tensor was bit-for-bit identical, ruling out
weight corruption or loading bugs.

### 6. Q/K position-by-position comparison

The query and key tensors entering SDPA were captured at each layer.  Position 0 was
identical between versions; positions 1–4 differed.  Because ROPE is only applied to
positions > 0 (it is the identity at position 0 by definition), this pointed squarely
at the ROPE computation as the source of the divergence.

### 7. ROPE cos/sin hook — smoking gun hypothesis

A hook on the return value of `OpensciRotaryEmbedding.forward` showed that in 5.x,
`cos[0, 1, :] = [1.0, 1.0, 1.0, 1.0]` and `sin[0, 1, :] = [0.0, 0.0, 0.0, 0.0]`
for position 1 — the identity rotation.  In 4.x the same positions held the expected
non-trivial values (~0.54, 0.80, 0.91, ...).  This confirmed that the ROPE was not
rotating anything after position 0 in 5.x.  At this stage the leading hypothesis was
a dtype rounding issue (`cos.to(dtype=x.dtype)` with bfloat16), but that turned out
to be wrong.

### 8. `rope_diag.py` — direct instrumentation of every intermediate

A targeted diagnostic script monkeypatched `OpensciRotaryEmbedding.forward` to write
every intermediate tensor to `/tmp/rope_diag_out.txt`, circumventing buffering and
progress-bar noise.  This revealed the actual root cause: `self.inv_freq[:4]` printed
as `[0.0, 0.0, 0.0, 0.0]` inside the forward method in 5.x, while 4.x showed
`[1.0, 0.649, 0.422, 0.274]`.  With all frequencies zero, `freqs = inv_freq × pos_id
= 0` for every position, hence `cos(0) = 1` and `sin(0) = 0` — the identity — for
all positions > 0.

---

## Final approach (Patch #5)

The root cause is transformers 5.x's **meta-device loading strategy**.  In 5.x,
`from_pretrained` initialises the model on the PyTorch `meta` device (a storage-less
virtual device) to avoid allocating memory twice.  It then materialises each weight
as it reads the checkpoint.  The `inv_freq` buffer in `OpensciRotaryEmbedding` is
registered with `persistent=False`, meaning it is intentionally excluded from the
`state_dict` so that it is never written to disk.  When 5.x materialises the model,
it finds no checkpoint entry for `inv_freq`, converts the meta tensor to a regular CPU
tensor, and fills it with zeros — the default for newly allocated storage.  The
`__init__` code that computed the correct frequencies already ran on the meta device
and its result was discarded.

The fix (Patch #5) adds a single guard at the top of `OpensciRotaryEmbedding.forward`:

```python
if not self.inv_freq.any():
    inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device=x.device)
    self.register_buffer("inv_freq", inv_freq.to(x.device), persistent=False)
```

On the first forward call under 5.x, `self.inv_freq` is all-zero, the condition fires,
and `rope_init_fn` (which was correctly set up in `__init__` and survived the
meta-device transition as a plain Python attribute, not a buffer) recomputes the
correct inverse frequencies on the real device.  Subsequent calls skip the branch
entirely.  After applying this patch and clearing the HuggingFace modules cache, the
logit for `" Paris"` is **13.0544** under transformers 4.48.0, 4.57.6, and 5.2.0 —
bit-for-bit identical across all three versions.  The full test suite (5 passed,
1 expected failure for the known 5.0/5.1 `generate()` regression) confirms the fix.
