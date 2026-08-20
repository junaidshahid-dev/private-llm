"""verify_checkpoint.py — GATE 3. Reload a checkpoint in a FRESH process and generate from it.

    python scripts/verify_checkpoint.py --checkpoint models/experiment-001/final

WHY A SEPARATE PROCESS

An in-process reload shares the interpreter with the code that just trained: the base model is
already resident, CUDA is initialised, the tokenizer is cached, module state is warm. A test
under those conditions can pass while a genuine restart fails — which is exactly the situation
you hit on Kaggle when the session dies and you come back tomorrow.

So this script assumes nothing. It loads the base model from scratch, applies only the saved
adapter, and generates. If this passes, the checkpoint is genuinely resumable.

It also verifies the provenance stamp, because a checkpoint whose manifest disagrees with the
current dataset or benchmark is not comparable to earlier results even if it loads fine.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# Same single-GPU pin as train.py, before torch is imported. Generation does not go through
# Trainer, so DataParallel is not the risk here — but a 4-bit model loaded with device_map={"":0}
# on a 2-GPU box invites the same class of device-mismatch confusion, and a verifier that runs
# under different device conditions than the trainer is not verifying the same thing.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROMPTS = [
    "Explain in one sentence why a scanner that produces false positives is worse than none.",
    "Write a Python function that returns the maximum drawdown of an equity curve.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default=None,
                    help="model lock/alias the adapter was trained on (default: the working base)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    ck = os.path.join(HERE, args.checkpoint)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("=" * 72)
    print(f"GATE 3 — fresh-process reload: {args.checkpoint}")
    print("=" * 72)

    check("checkpoint directory exists", os.path.isdir(ck), ck)
    if fails:
        return 1
    files = os.listdir(ck)
    check("adapter weights present", "adapter_model.safetensors" in files)
    check("adapter config present", "adapter_config.json" in files)
    check("run manifest present", "run_manifest.json" in files)
    if fails:
        return 1

    prov = json.load(open(os.path.join(ck, "run_manifest.json"), encoding="utf-8"))
    from serving.model_spec import load_lock          # match the base the adapter was trained on
    lock = load_lock(args.model)
    check("manifest model revision matches lockfile",
          prov["model_revision"] == lock["revision"], prov["model_revision"][:12])

    bench_lock = os.path.join(HERE, "evaluation", "frozen", "benchmark_v1", "benchmark.lock.json")
    if os.path.exists(bench_lock):
        cur = json.load(open(bench_lock, encoding="utf-8"))["sha256"]
        same = prov.get("benchmark_sha256") == cur
        check("benchmark hash unchanged since training", same,
              "results comparable" if same else
              "BENCHMARK CHANGED — this checkpoint's scores are not comparable to new ones")

    adapter_cfg = json.load(open(os.path.join(ck, "adapter_config.json"), encoding="utf-8"))
    check("adapter targets match lockfile",
          set(adapter_cfg["target_modules"]) == set(lock["lora_targets_validated"]),
          str(sorted(adapter_cfg["target_modules"])))

    if not torch.cuda.is_available():
        print("\n  no CUDA — structural checks passed, generation skipped.")
        print("  Run this on the GPU box; that is the half that matters.")
        return 1 if fails else 0

    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=q["double_quant"],
                            bnb_4bit_compute_dtype=torch.float16)
    print("\n  loading base model from scratch (nothing cached in this process)...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
    check("base model loads cold", True, f"{time.time()-t0:.0f}s")

    model = PeftModel.from_pretrained(base, ck)
    model.eval()
    check("adapter applies to cold base", True)

    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\n  generating:")
    ok_gen = True
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt").to(0)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        print(f"\n    > {p[:64]}")
        print(f"      {text[:220]}")
        if not text:
            ok_gen = False
    check("generation produced non-empty output", ok_gen)
    check("peak VRAM during inference recorded", True,
          f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    print("\n" + "=" * 72)
    if fails:
        print(f"GATE 3 FAILED: {fails}")
        return 1
    print("GATE 3 PASSED — checkpoint reloads cold and generates.")
    print("All four gates are now green. The real run is safe to start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
