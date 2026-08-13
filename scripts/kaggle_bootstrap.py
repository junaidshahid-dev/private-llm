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

# Everything floats to latest EXCEPT transformers, which is capped for a measured reason:
# 5.0 moved DeepSeek-V3 MoE experts from ModuleList[Linear] to 3D nn.Parameter tensors, which
# bitsandbytes cannot quantise. That takes the model from 8.49GB (97.9% quantisable) to 30.08GB
# (7.7%) — it stops fitting any free GPU. 4.57.6 is the newest 4.x release; there is nothing
# between it and 5.0. Verified working together: peft 0.20, accelerate 1.14, datasets 5.0.1,
# torch 2.13. Re-check with scripts/check_quantizable.py after any upgrade.
PINS = [
    "transformers==4.57.6",     # CAP — see scripts/check_quantizable.py for the measurement
    "peft>=0.20",
    "bitsandbytes>=0.50",
    "accelerate>=1.14",
    "datasets>=5.0",
    "trl>=1.10",
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
        n = torch.cuda.device_count()
        if n > 1:
            print(f"[note] {n} GPUs visible — train.py pins itself to GPU 0. HF Trainer would")
            print("       otherwise use DataParallel, which corrupts bitsandbytes quant_state")
            print("       and fails as CUBLAS_STATUS_EXECUTION_FAILED inside q_proj.")
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
# /kaggle/working is capped around 20GB because it is persisted as notebook OUTPUT. The model
# download does NOT go there — it goes to the Hugging Face cache. So measure every mount and
# put HF_HOME on whichever one can actually hold ~32GB of bf16 shards.
CANDIDATES = ["/kaggle/temp", "/tmp", "/root", "/kaggle/working", "."]
print("\ndisk by mount:")
best, best_free = None, 0
for p in CANDIDATES:
    if not os.path.isdir(p):
        continue
    try:
        u = shutil.disk_usage(p)
    except OSError:
        continue
    print(f"    {p:18} {u.free/1e9:6.1f} GB free of {u.total/1e9:6.1f} GB")
    # /kaggle/working is output-capped regardless of what the filesystem reports
    usable = u.free if p != "/kaggle/working" else min(u.free, 20e9)
    if usable > best_free:
        best, best_free = p, usable

check("a mount with >= 40GB free exists", best_free / 1e9 >= 40,
      f"best: {best} with {best_free/1e9:.0f} GB — need ~32GB for the bf16 shards")

if best and best_free / 1e9 >= 40:
    cache = os.path.join(best, "hf_cache")
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = cache
    with open("/kaggle/working/.hf_env", "w") as f:
        f.write(f"export HF_HOME={cache}\nexport HF_HUB_CACHE={cache}\n")
    print(f"    -> HF_HOME set to {cache}")
    print(f"       In LATER cells this must be set again, because each ! cell is a new shell:")
    print(f"       import os; os.environ['HF_HOME']='{cache}'")

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

# ---- conflicting preinstalled packages --------------------------------------
# Kaggle ships torchao 0.10.0. peft's is_torchao_available() RAISES ImportError when it finds a
# version below 0.16 rather than returning False, so LoRA attach dies inside _create_new_module
# even though we never use torchao — quantisation here is bitsandbytes.
#
# Removing it is safer than upgrading: torchao pins torch versions, and torch is the one thing
# on this image we do not want to disturb.
try:
    import importlib.metadata as _md
    tv = _md.version("torchao")
    major_minor = tuple(int(x) for x in tv.split(".")[:2])
    if major_minor < (0, 16):
        print(f"\nremoving torchao {tv} — peft raises on versions below 0.16, and nothing here")
        print("uses it (quantisation is bitsandbytes)")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                       check=False)
    else:
        print(f"\ntorchao {tv} — new enough, leaving it")
except Exception:                                            # noqa: BLE001
    pass                                                     # not installed: nothing to do

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
