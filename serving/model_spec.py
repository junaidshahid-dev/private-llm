"""model_spec.py — the ONE place the stack decides which model to load.

    from serving.model_spec import load_lock
    lock = load_lock()          # default: MODEL_SPEC.lock.json (the frozen Moonlight baseline)

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
DEFAULT_LOCK = "MODEL_SPEC.lock.json"


def lock_path(explicit: str | None = None) -> str:
    """Resolve the lock file: explicit arg > MODEL_LOCK env > the frozen baseline. Relative names
    are resolved against the repo root."""
    name = explicit or os.environ.get("MODEL_LOCK") or DEFAULT_LOCK
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def load_lock(explicit: str | None = None) -> dict:
    path = lock_path(explicit)
    if not os.path.exists(path):
        raise FileNotFoundError(f"model lock not found: {path} "
                                "(set MODEL_LOCK to a lock file's name, or pass --model-lock)")
    spec = json.load(open(path, encoding="utf-8"))
    spec["_lock_path"] = path
    return spec
