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

After running this script, the fixed model loads cleanly with transformers 5.2+
without any runtime monkey-patching.

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
