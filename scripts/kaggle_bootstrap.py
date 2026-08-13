"""kaggle_bootstrap.py — paste this as the FIRST cell of a Kaggle notebook.

It checks the environment before anything expensive happens, installs the pinned versions, and
refuses to continue if the session cannot succeed. A failed check here costs seconds; the same
failure after a 32GB download costs an hour of a 9-hour session.

    # Kaggle cell 1
    !git clone <your repo> /kaggle/working/LLM   ||  upload the folder as a Dataset
    %cd /kaggle/working/LLM
    !python scripts/kaggle_bootstrap.py

CHECKS, in the order that matters:

  GPU present            no GPU means Accelerator is off in the right-hand panel
  compute capability     T4 = 7.5 OK.  P100 = 6.0 CANNOT run bitsandbytes 4-bit.
                         Kaggle assigns either at random. Restart the session to reroll.
  internet               needed for the model download; requires phone verification
  disk                   the bf16 download is ~32GB even though it loads as ~8.5GB in VRAM
  pinned packages        transformers 4.57.6 - NOT latest, see MODEL_SPEC.lock.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

PINS = [
    "transformers==4.57.6",     # 5.x breaks; see the lockfile for why
    "peft>=0.11",
    "bitsandbytes>=0.43",
    "accelerate>=0.30",
    "datasets>=2.19",
    "trl>=0.9",
    "tiktoken",                 # the tokenizer is tiktoken-based, not sentencepiece
    "blobfile",
    "pyyaml",
]

fail = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fail.append(name)


print("=" * 68)
print("KAGGLE PRE-FLIGHT")
print("=" * 68)

# ---- GPU -------------------------------------------------------------------
try:
    import torch
    has = torch.cuda.is_available()
    check("GPU visible", has, torch.cuda.get_device_name(0) if has else
          "turn on Accelerator -> GPU T4 x2 in the right-hand panel")
    if has:
        cap = torch.cuda.get_device_capability(0)
        cc = cap[0] * 10 + cap[1]
        ok = cc >= 75
        check("compute capability >= 7.5", ok,
              f"{cap[0]}.{cap[1]} — " + ("T4, good" if ok else
              "this is a P100. bitsandbytes 4-bit will NOT work. "
              "Stop the session and start a new one to get a T4."))
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("VRAM >= 15GB", total >= 15, f"{total:.1f} GB")
        check("bf16 support", torch.cuda.is_bf16_supported(),
              "expected False on T4 — config already uses fp16")
except ImportError:
    check("torch importable", False, "no torch in this environment")

# ---- internet ---------------------------------------------------------------
try:
    import urllib.request
    urllib.request.urlopen("https://huggingface.co", timeout=15)
    check("internet reachable", True, "huggingface.co")
except Exception as e:                                       # noqa: BLE001
    check("internet reachable", False,
          f"{type(e).__name__} — switch Internet ON (needs phone verification)")

# ---- disk -------------------------------------------------------------------
free = shutil.disk_usage("/kaggle/working" if os.path.isdir("/kaggle/working") else ".").free
check("free disk >= 40GB", free / 1e9 >= 40,
      f"{free/1e9:.0f} GB free — the bf16 download alone is ~32GB")

# ---- repo -------------------------------------------------------------------
lock_p = "MODEL_SPEC.lock.json"
check("MODEL_SPEC.lock.json present", os.path.exists(lock_p),
      "run from the repo root")
if os.path.exists(lock_p):
    lock = json.load(open(lock_p, encoding="utf-8"))
    check("revision pinned", len(lock["revision"]) == 40, lock["revision"][:12])
check("frozen benchmark present",
      os.path.exists("evaluation/frozen/benchmark_v1/benchmark.lock.json"))
check("training data present", os.path.exists("data/train/train.jsonl"))

if fail:
    print("\n" + "=" * 68)
    print(f"STOP — {len(fail)} check(s) failed: {fail}")
    print("Fix these before installing anything. Nothing below is worth the session time.")
    sys.exit(1)

# ---- install ----------------------------------------------------------------
print("\ninstalling pinned versions (transformers is deliberately NOT latest)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)

import importlib
importlib.invalidate_caches()
import transformers
print(f"\ntransformers {transformers.__version__}")
if transformers.__version__ != "4.57.6":
    print("  WARNING: version drift. 5.x removes is_torch_fx_available and the DeepSeek-V3")
    print("  path will fail. Pin it before training.")

print("\n" + "=" * 68)
print("PRE-FLIGHT PASSED. Next:")
print("  !python scripts/smoke_test.py                    # ~1 min, no download")
print("  !python training/train.py --pilot                # downloads ~32GB, then 10 steps")
print("  !python scripts/verify_checkpoint.py --checkpoint models/experiment-001/final")
print("=" * 68)
