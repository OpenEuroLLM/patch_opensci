"""
hotfix_opensci.py

Copies a model directory to <src_dir>_fixed and patches every modeling_*.py
for compatibility with transformers 5.0+:

  1. LossKwargs → TransformersKwargs          (LossKwargs removed in 5.0)
  2. ROPE 'default' fallback inlined           ('default' key removed from
                                                ROPE_INIT_FUNCTIONS in 5.0)
  3. _tied_weights_keys list → dict           (5.0 requires a {key: source} dict;
                                               list format leaves lm_head
                                               uninitialised at load time)
  4. SDPA dispatch inlined                    (transformers 5.x passes enable_gqa=True
                                               to scaled_dot_product_attention even for
                                               MHA models, selecting a different numerical
                                               path than 4.x; inlining the 4.x wrapper
                                               restores identical attention computation)
  5. inv_freq re-init guard added             (transformers 5.x uses meta-device loading:
                                               non-persistent buffers are zeroed when weights
                                               are materialised; the guard re-computes
                                               inv_freq on the first forward pass if zero,
                                               restoring correct ROPE position encodings)

After running this script, the fixed model loads cleanly with transformers 5.2+
without any runtime monkey-patching, and produces logits identical to 4.x.

Usage:
    uv run python hotfix_opensci.py --src_dir <model_path>
"""

import argparse
import re
import shutil
from pathlib import Path

# Inline fallback inserted when ROPE_INIT_FUNCTIONS["default"] is missing.
_ROPE_FALLBACK = '''\
if self.rope_type in ROPE_INIT_FUNCTIONS:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        else:
            # 'default' rope type was removed from ROPE_INIT_FUNCTIONS in transformers 5.0;
            # inline the standard computation as a fallback.
            def _default_rope_init(config, device=None, **kwargs):
                base = config.rope_theta
                head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
                return (
                    1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)),
                    1.0,
                )
            self.rope_init_fn = _default_rope_init\
'''


# Replacement for the SDPA dispatch block that inlines the 4.x wrapper behaviour.
# transformers 5.x adds enable_gqa=True to scaled_dot_product_attention even for
# plain MHA (groups=1), which routes to a different PyTorch kernel and shifts logits
# by ~0.6 relative to 4.x. Inlining the 4.x wrapper avoids this.
_SDPA_OLD = '''\
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]\
'''

_SDPA_STABLE = '''\
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation == "sdpa" and not kwargs.get("output_attentions", False):
            # Inline the 4.x-compatible SDPA wrapper: never passes enable_gqa so the
            # same PyTorch kernel is selected regardless of transformers version.
            def attention_interface(module, query, key, value, attention_mask, scaling, dropout=0.0, is_causal=None, **kw):
                if hasattr(module, "num_key_value_groups"):
                    key = repeat_kv(key, module.num_key_value_groups)
                    value = repeat_kv(value, module.num_key_value_groups)
                query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
                if is_causal is None:
                    is_causal = attention_mask is None and query.shape[2] > 1
                out = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask,
                    dropout_p=dropout, scale=scaling, is_causal=is_causal,
                )
                return out.transpose(1, 2).contiguous(), None
        elif self.config._attn_implementation not in ("eager", "sdpa"):
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]\
'''


# Guard inserted into OpensciRotaryEmbedding.forward to re-initialise inv_freq
# if it was zeroed by transformers 5.x meta-device loading.
# Non-persistent buffers are not saved in the checkpoint; when 5.x materialises
# the model from the meta device, those buffers become zero tensors.  Detecting
# this on the first forward pass and recomputing from rope_init_fn restores the
# correct ROPE frequencies without any change to the __init__ signature.
_ROPE_REINIT_OLD = '''\
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)\
'''

_ROPE_REINIT_FIXED = '''\
        # Re-initialize inv_freq if it was zeroed during meta-device loading (transformers 5.x).
        # Non-persistent buffers are not stored in the checkpoint; 5.x materialises them as
        # zeros when moving the model off the meta device.
        if not self.inv_freq.any():
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device=x.device)
            self.register_buffer("inv_freq", inv_freq.to(x.device), persistent=False)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)\
'''


def patch_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    original = src

    # 1. LossKwargs → TransformersKwargs
    src = src.replace("LossKwargs", "TransformersKwargs")

    # 2. ROPE 'default' fallback
    #    Replace: self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
    #    With a try/else that inlines the default computation.
    src = re.sub(
        r"self\.rope_init_fn\s*=\s*ROPE_INIT_FUNCTIONS\[self\.rope_type\]",
        _ROPE_FALLBACK,
        src,
    )

    # 3. _tied_weights_keys: list → dict
    #    e.g. ["lm_head.weight"]  →  {"lm_head.weight": "model.embed_tokens.weight"}
    def list_to_dict(m: re.Match) -> str:
        keys = re.findall(r'"([^"]+)"', m.group(1))
        pairs = ", ".join(f'"{k}": "model.embed_tokens.weight"' for k in keys)
        return f"_tied_weights_keys = {{{pairs}}}"

    src = re.sub(
        r"_tied_weights_keys\s*=\s*(\[[^\]]*\])",
        list_to_dict,
        src,
    )

    # 4. SDPA dispatch: replace with inline wrapper that omits enable_gqa
    src = src.replace(_SDPA_OLD, _SDPA_STABLE)

    # 5. inv_freq re-init guard: re-compute if zeroed by meta-device loading
    src = src.replace(_ROPE_REINIT_OLD, _ROPE_REINIT_FIXED)

    if src == original:
        return False
    path.write_text(src, encoding="utf-8")
    return True


def hotfix_opensci(src_dir: str | Path) -> Path:
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {src_dir}")

    fixed_dir = src_dir.parent / (src_dir.name + "_fixed")
    print(f"Copying '{src_dir}' -> '{fixed_dir}' ...")
    if fixed_dir.exists():
        shutil.rmtree(fixed_dir)
    shutil.copytree(src_dir, fixed_dir)

    patched = 0
    for model_file in sorted(fixed_dir.glob("modeling_*.py")):
        if patch_file(model_file):
            print(f"  Patched: {model_file.name}")
            patched += 1
        else:
            print(f"  Skipped (nothing to fix): {model_file.name}")

    print(f"Done. {patched} file(s) patched. Fixed model at '{fixed_dir}'.")
    return fixed_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patch open-sci model files for transformers 5.0+ compatibility."
    )
    parser.add_argument("--src_dir", required=True, help="Path to the model directory.")
    args = parser.parse_args()
    hotfix_opensci(args.src_dir)
