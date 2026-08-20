"""model_spec.py — the ONE place the stack decides which model to load.

    from serving.model_spec import load_lock
    lock = load_lock()          # default: the working base (Qwen2.5-Coder-14B)
    lock = load_lock("moonlight")   # the frozen Moonlight baseline, still reachable by alias

The architecture is deliberately model-agnostic: RAG, MCP, verification, the lab, and the benchmarks
never name a model — they call load_lock(). To A/B a STRONGER base model against the frozen Moonlight
baseline while keeping everything else identical, you do NOT edit the baseline. You:

    1. write a second lock file, e.g. MODEL_SPEC.qwen.lock.json (same shape as MODEL_SPEC.lock.json:
       model, revision, context_length, quantization),
    2. point the run at it:  MODEL_LOCK=MODEL_SPEC.qwen.lock.json python evaluation/run_secv3.py ...
       (or pass --model-lock where the runner exposes it),
    3. compare the results to the baseline run.

That answers the experiment's question — "does a stronger brain improve security reasoning when
everything else is identical?" — without throwing away any of the work, and without unfreezing the
Moonlight baseline.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The WORKING base is Qwen2.5-Coder-14B, chosen after the held-out benchmark decided it (sec-v4
# 0.831 vs Moonlight 0.787, equal injection resistance, ~3x faster). The Moonlight baseline lock
# (MODEL_SPEC.lock.json) is kept frozen and reachable as `moonlight`/`baseline` for reproducibility;
# it is not rewritten. Override per-run with MODEL_LOCK=... or --model.
DEFAULT_LOCK = "MODEL_SPEC.qwen25-coder-14b.lock.json"

# Friendly names -> lock files, so `--model qwen` works. Unknown names fall back to
# MODEL_SPEC.<name>.lock.json, and an explicit *.lock.json path is used as-is.
ALIASES = {
    "moonlight": "MODEL_SPEC.lock.json",
    "baseline": "MODEL_SPEC.lock.json",
    "qwen": "MODEL_SPEC.qwen25-coder-14b.lock.json",
    "qwen-coder": "MODEL_SPEC.qwen25-coder-14b.lock.json",
    "qwen-14b": "MODEL_SPEC.qwen25-14b.lock.json",
}


def resolve(name: str | None) -> str | None:
    """A friendly --model value -> a lock file name (not a path). None stays None."""
    if not name:
        return None
    if name in ALIASES:
        return ALIASES[name]
    if name.endswith(".lock.json"):
        return name
    return f"MODEL_SPEC.{name}.lock.json"


def lock_path(explicit: str | None = None) -> str:
    """Resolve the lock file: explicit (alias/name/path) > MODEL_LOCK env > the frozen baseline.
    Relative names are resolved against the repo root."""
    name = resolve(explicit) or os.environ.get("MODEL_LOCK") or DEFAULT_LOCK
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def load_lock(explicit: str | None = None) -> dict:
    path = lock_path(explicit)
    if not os.path.exists(path):
        raise FileNotFoundError(f"model lock not found: {path} "
                                "(set MODEL_LOCK to a lock file's name, or pass --model-lock)")
    spec = json.load(open(path, encoding="utf-8"))
    spec["_lock_path"] = path
    return spec
