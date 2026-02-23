"""
Inference script for open-sci model.

For transformers 5.2+, run against the hotfix-patched model (no runtime patches needed):
    uv run python hotfix_opensci.py --src_dir <model_path>
    uv run python inference.py --model_path <model_path>_fixed

For transformers 4.x, the original model directory works directly:
    uv run python inference.py --model_path <model_path>
"""
import argparse

import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer

_tv = Version(transformers.__version__)

# TransformersKwargs was introduced in 5.0 (renamed from LossKwargs).
# Back-fill it for 4.x so the model's remote code can import it.
if _tv < Version("5.0.0"):
    import transformers.utils as _tu
    if not hasattr(_tu, "TransformersKwargs"):
        _tu.TransformersKwargs = _tu.LossKwargs

# torch_dtype was renamed to dtype in transformers 5.0
_dtype_key = "dtype" if _tv >= Version("5.0.0") else "torch_dtype"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

print(f"transformers {transformers.__version__}")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    **{_dtype_key: torch.float16},
)
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\nPrompt: {prompt!r}")
print("Generating...")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Completion: {completion!r}")
