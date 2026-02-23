"""
Tests that inference.py produces a sensible completion across transformers versions.

Each test spawns a subprocess:
    uv run [--with transformers==X] python inference.py --model_path ...

4.x tests use the original model directory (TransformersKwargs back-filled at
runtime). 5.x tests use the hotfix_opensci.py output (_fixed directory) which
has all patches baked into the model files — no runtime monkey-patching needed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODEL_ORIG   = str(Path(__file__).parent / "open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384")
MODEL_FIXED  = str(Path(__file__).parent / "open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384_fixed")
INFERENCE    = str(Path(__file__).parent / "inference.py")
LOGIT_CHECK  = str(Path(__file__).parent / "logit_check.py")
EXPECTED_WORD = "Paris"

UV = shutil.which("uv") or "uv"


def _run(transformers_version: str, model_path: str) -> subprocess.CompletedProcess:
    cmd = [UV, "run", "--with", f"transformers=={transformers_version}",
           "python", INFERENCE, "--model_path", model_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # pytest's default fd-level capture redirects even sys.__stdout__, so we
    # write directly to /dev/tty (the controlling terminal) to always be visible.
    try:
        with open("/dev/tty", "w") as tty:
            if result.stdout:
                tty.write(result.stdout)
            if result.stderr:
                tty.write(result.stderr)
            tty.flush()
    except OSError:
        pass  # not a real terminal (CI / redirected output)
    return result


def _assert_ok(result: subprocess.CompletedProcess, label: str):
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"[{label}] exited with code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert EXPECTED_WORD in combined, (
        f"[{label}] Expected '{EXPECTED_WORD}' in output, got:\n{combined}"
    )


# ── 4.x: original model, TransformersKwargs back-filled at runtime ────────────

@pytest.mark.parametrize("version", ["4.48.0", "4.49.0", "4.57.6"])
def test_inference_4x(version):
    """4.x: original model directory, inference.py handles TransformersKwargs."""
    _assert_ok(_run(version, MODEL_ORIG), version)


# ── 5.x: hotfix-patched model, no runtime patching needed ────────────────────

@pytest.mark.parametrize("version", [
    pytest.param("5.0.0", marks=pytest.mark.xfail(
        reason="transformers 5.0/5.1 has a known generate() regression unrelated "
               "to our patches; fixed in 5.2.0",
        strict=True,
    )),
    "5.2.0",
])
def test_inference_5x(version):
    """5.x: hotfix-patched model, clean inference without any monkey-patching."""
    _assert_ok(_run(version, MODEL_FIXED), version)


# ── logit consistency: same argmax across all supported versions ──────────────

def _run_logits(version: str, model_path: str) -> list[dict]:
    """Run logit_check.py and return the top_tokens list."""
    cmd = [UV, "run", "--with", f"transformers=={version}",
           "python", LOGIT_CHECK, "--model_path", model_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    try:
        with open("/dev/tty", "w") as tty:
            if result.stderr:
                tty.write(result.stderr)
            tty.flush()
    except OSError:
        pass
    assert result.returncode == 0, (
        f"[{version}] logit_check exited {result.returncode}\n"
        f"STDERR:\n{result.stderr}"
    )
    return json.loads(result.stdout)["top_tokens"]


def test_logits_top_token_consistent():
    """
    The most-likely next token must be identical across all supported versions.

    Logit *values* may drift slightly due to attention-backend differences
    (eager / SDPA) between transformers versions, but the argmax — which drives
    greedy decoding — should agree everywhere.
    """
    cases = [
        ("4.48.0", MODEL_ORIG),
        ("4.57.6", MODEL_ORIG),
        ("5.2.0",  MODEL_FIXED),
    ]
    top_tokens: dict[str, dict] = {}
    for version, model_path in cases:
        tokens = _run_logits(version, model_path)
        top_tokens[version] = tokens[0]
        try:
            with open("/dev/tty", "w") as tty:
                tty.write(
                    f"[{version}] top token: {tokens[0]['token']!r:20s} "
                    f"logit={tokens[0]['logit']:.4f}\n"
                )
                tty.flush()
        except OSError:
            pass

    top_ids = {v: t["token_id"] for v, t in top_tokens.items()}
    assert len(set(top_ids.values())) == 1, (
        "Top predicted token differs across versions:\n"
        + "\n".join(
            f"  {v}: {t['token']!r} (id={t['token_id']}, logit={t['logit']})"
            for v, t in top_tokens.items()
        )
    )
