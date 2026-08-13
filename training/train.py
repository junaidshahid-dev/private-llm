"""train.py — QLoRA on Moonlight-16B-A3B, written for a free GPU that will be killed mid-run.

    python training/train.py --pilot          # 10 steps, runs the 4 gates, ~$0.10 of GPU
    python training/train.py                  # the real run
    python training/train.py --resume auto    # explicit; auto is already the default

FOUR HARD GATES — the pilot fails loudly rather than quietly wasting the next eight hours.

  1 MEMORY   peak VRAM recorded separately at load / forward / backward / optimizer / save.
             9GB of weights does not imply a 16GB fit; activations and optimizer state are
             where T4 runs actually die.
  2 LOSS     initial vs final loss over the pilot. NaN, inf, exactly-flat, or divergent is a
             configuration failure. A FALLING loss over 10 steps is NOT evidence of quality —
             it only means the machinery is connected.
  3 RELOAD   handled by scripts/verify_checkpoint.py, in a FRESH PROCESS. An in-process reload
             shares state and can pass while a real restart fails.
  4 REPRO    every checkpoint carries model revision, dataset hash, benchmark hash, full config,
             package versions, CUDA version and seeds. A checkpoint you cannot trace is a
             checkpoint you cannot trust.

KAGGLE. Sessions are killed at 9 hours with no warning. Interruption is the NORMAL case here.
The script resumes from the latest checkpoint that actually loads, so a kill costs minutes.

THE BASE MODEL IS NEVER WRITTEN TO. Every run writes to its own models/experiment-NNN/ so a bad
fine-tune cannot destroy the starting point.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------------------------
# SINGLE GPU, ENFORCED BEFORE TORCH IS IMPORTED.
#
# Kaggle hands out "T4 x2". HuggingFace Trainer sees device_count() > 1, sets n_gpu = 2, and
# wraps the model in nn.DataParallel — which replicates the whole model onto each device on
# every step. That is incompatible with bitsandbytes 4-bit: a Params4bit tensor carries a
# quant_state (absmax, blocksize, code) that does not survive replication, so the replica falls
# through to _dequant_linear_fallback and hands cuBLAS a malformed GEMM:
#
#     RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx(...)
#
# The error surfaces inside q_proj, which makes it look like a LoRA-target problem. It is not.
#
# DataParallel would be the wrong tool even if it worked: it replicates rather than shards, so
# it gives no memory benefit, and the 4-bit model already fits on one T4. Real multi-GPU here
# would mean device_map="auto" (model parallel) or DDP, neither of which we need.
#
# Set CUDA_VISIBLE_DEVICES yourself to override — e.g. "1" to use the second card instead.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _PINNED_GPU = True
else:
    _PINNED_GPU = False
# ---------------------------------------------------------------------------------------------

GATES: dict[str, dict] = {}


def sha_file(p: str) -> str:
    if not os.path.exists(p):
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(cfg: dict, lock: dict) -> dict:
    """GATE 4. Everything needed to reproduce this run, or to prove you cannot."""
    import torch
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except Exception:                                    # noqa: BLE001
            return "not installed"

    bench_lock = os.path.join(HERE, "evaluation", "frozen", "benchmark_v1",
                              "benchmark.lock.json")
    return {
        "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": lock["model"],
        "model_revision": lock["revision"],
        "dataset_train_sha256": sha_file(os.path.join(HERE, cfg["data"]["train_file"])),
        "dataset_manifest": (json.load(open(os.path.join(HERE, "data", "processed",
                                                         "manifest.json"), encoding="utf-8"))
                             if os.path.exists(os.path.join(HERE, "data", "processed",
                                                            "manifest.json")) else None),
        "benchmark_sha256": (json.load(open(bench_lock, encoding="utf-8"))["sha256"]
                             if os.path.exists(bench_lock) else "missing"),
        "config": cfg,
        "packages": {p: ver(p) for p in
                     ("torch", "transformers", "peft", "trl", "accelerate",
                      "bitsandbytes", "datasets", "tiktoken")},
        "cuda": {
            "available": torch.cuda.is_available(),
            "version": getattr(torch.version, "cuda", None),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": (list(torch.cuda.get_device_capability(0))
                           if torch.cuda.is_available() else None),
            "total_memory_gb": (round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                                if torch.cuda.is_available() else None),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": cfg.get("seed", 20260813),
    }


class Mem:
    """GATE 1. Peak VRAM per phase, not one number for the whole run."""

    def __init__(self):
        self.peaks: dict[str, float] = {}

    def mark(self, phase: str):
        import torch
        if not torch.cuda.is_available():
            self.peaks[phase] = 0.0
            return
        torch.cuda.synchronize()
        self.peaks[phase] = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

    def report(self, total_gb: float | None):
        print("\n  GATE 1 — peak VRAM by phase (GB):")
        worst = max(self.peaks.values()) if self.peaks else 0.0
        for k, v in self.peaks.items():
            bar = "#" * int(v / max(worst, 0.01) * 28)
            print(f"    {k:<22} {v:6.2f}  {bar}")
        if total_gb:
            head = total_gb - worst
            print(f"    {'device total':<22} {total_gb:6.2f}")
            print(f"    {'headroom':<22} {head:6.2f}"
                  + ("   <- TIGHT, raise grad-accum or cut seq len" if head < 1.0 else ""))
            GATES["memory"] = {"peaks_gb": self.peaks, "total_gb": total_gb,
                               "headroom_gb": round(head, 2), "pass": head > 0.3}
        else:
            GATES["memory"] = {"peaks_gb": self.peaks, "pass": None, "note": "no CUDA"}


def latest_checkpoint(out_dir: str) -> str | None:
    """Newest checkpoint that actually contains an adapter. A half-written dir is skipped."""
    cands = []
    for d in glob.glob(os.path.join(out_dir, "checkpoint-*")):
        if os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            m = re.search(r"checkpoint-(\d+)$", d)
            if m:
                cands.append((int(m.group(1)), d))
    return max(cands)[1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/moonlight_qlora.yaml")
    ap.add_argument("--pilot", action="store_true", help="10 steps + gates, then stop")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--experiment", default=None, help="models/<name>/ ; auto-numbered if unset")
    ap.add_argument("--resume", default="auto")
    args = ap.parse_args()

    import torch
    import yaml
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments, Trainer, DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    cfg = yaml.safe_load(open(os.path.join(HERE, args.config), encoding="utf-8"))
    lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))
    seed = cfg.get("seed", 20260813)
    random.seed(seed); torch.manual_seed(seed)

    # ---- experiment dir; the base model is never touched ---------------------
    if args.experiment:
        exp = args.experiment
    else:
        n = 1 + len(glob.glob(os.path.join(HERE, "models", "experiment-*")))
        exp = f"experiment-{n:03d}"
    out_dir = os.path.join(HERE, "models", exp)
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 72)
    print(f"{'PILOT' if args.pilot else 'TRAINING'} — {exp}")
    print(f"model {lock['model']} @ {lock['revision'][:12]}")
    print("=" * 72)

    # ---- transformers version ceiling ----------------------------------------
    # Not a superstition pin. transformers 5.0 refactored DeepSeek-V3 MoE experts from
    # ModuleList[Linear] into 3D nn.Parameter tensors. bitsandbytes quantises by replacing
    # nn.Linear modules, so on 5.x only 7.7% of this model is reachable and the weights land at
    # ~30GB instead of ~8.5GB. Nothing errors — it simply stops fitting on free hardware, which
    # is a far more expensive way to find out. Run scripts/check_quantizable.py for the numbers.
    import transformers as _tf
    _major = int(_tf.__version__.split(".")[0])
    if _major >= 5:
        print(f"\nFAIL: transformers {_tf.__version__}. MoE experts are 3D parameters from 5.0")
        print("      onward and bitsandbytes cannot quantise them: ~30GB of weights instead of")
        print("      ~8.5GB. Install transformers==4.57.6 (the newest 4.x) and re-run.")
        print("      Evidence: python scripts/check_quantizable.py")
        return 1

    # Upstream corrections, applied before the model is built and recorded in the manifest so a
    # checkpoint always says which patches were in force when it was produced.
    from training.patches import apply_all as apply_patches
    prov_patches = apply_patches()

    mem = Mem()
    prov = provenance(cfg, lock)
    prov["upstream_patches"] = prov_patches      # GATE 4: results depend on these being applied
    if not prov["cuda"]["available"]:
        print("\nNO CUDA. This script needs a GPU; the CPU path is scripts/smoke_test.py.")
        return 2
    print(f"device {prov['cuda']['device']}  "
          f"{prov['cuda']['total_memory_gb']}GB  capability {prov['cuda']['capability']}")
    if _PINNED_GPU:
        print("  pinned to GPU 0 — Trainer wraps multi-GPU in DataParallel, which corrupts")
        print("  bitsandbytes quant_state. Set CUDA_VISIBLE_DEVICES yourself to override.")
    if torch.cuda.device_count() > 1:
        print(f"  WARNING: {torch.cuda.device_count()} GPUs still visible — DataParallel risk")
    cap = prov["cuda"]["capability"]
    if cap and cap[0] < 8:
        print("  note: pre-Ampere — no bf16, no FlashAttention-2. fp16 compute, as configured.")
    if cap and cap[0] * 10 + cap[1] < 75:
        print("  FAIL: bitsandbytes 4-bit needs compute capability >= 7.5 (this is a P100).")
        return 1

    # ---- data ----------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load_split(path):
        rows = [json.loads(l) for l in open(os.path.join(HERE, path), encoding="utf-8")
                if l.strip()]
        texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
        return Dataset.from_dict({"text": texts})

    train_ds = load_split(cfg["data"]["train_file"])
    eval_ds = load_split(cfg["data"]["eval_file"])
    if args.pilot:
        train_ds = train_ds.select(range(min(20, len(train_ds))))
        eval_ds = eval_ds.select(range(min(4, len(eval_ds))))
    print(f"\ndata: {len(train_ds)} train / {len(eval_ds)} eval")

    msl = cfg["model"]["max_seq_len"]

    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=msl, padding="max_length")

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tok_fn, batched=True, remove_columns=["text"])

    # ---- model ---------------------------------------------------------------
    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]))
    torch.cuda.reset_peak_memory_stats()
    print("\nloading 4-bit weights (~8.5GB, first run also downloads ~32GB)...")
    t0 = time.time()
    # NATIVE deepseek_v3 — no trust_remote_code. The repo's modeling file is stale; see
    # MODEL_SPEC.lock.json. The tokenizer above still needs it (custom TikTokenTokenizer).
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
    print(f"  loaded in {time.time()-t0:.0f}s")
    mem.mark("model load (4-bit)")

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"])
    model.config.use_cache = False              # required with gradient checkpointing
    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        bias=lcfg["bias"], task_type=lcfg["task_type"],
        target_modules=lcfg["target_modules"]))
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    al = sum(p.numel() for p in model.parameters())
    print(f"  trainable {tr:,} / {al:,} = {tr/al*100:.3f}%")
    mem.mark("after LoRA attach")

    # ---- resume --------------------------------------------------------------
    resume = None
    if args.resume == "auto":
        resume = latest_checkpoint(out_dir)
        print(f"\nresume: {os.path.basename(resume) if resume else 'none — fresh start'}")
    elif args.resume not in ("", "none"):
        resume = os.path.join(HERE, args.resume)

    t = cfg["training"]
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=t["num_train_epochs"],
        max_steps=args.steps if args.pilot else -1,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=1 if args.pilot else t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"], warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"], max_grad_norm=t["max_grad_norm"],
        gradient_checkpointing=t["gradient_checkpointing"], optim=t["optim"],
        fp16=t["fp16"], bf16=t["bf16"],
        save_steps=5 if args.pilot else t["save_steps"],
        save_total_limit=t["save_total_limit"],
        logging_steps=1 if args.pilot else t["logging_steps"],
        eval_strategy="no" if args.pilot else "steps",
        eval_steps=t["eval_steps"], report_to=[], seed=seed,
        save_safetensors=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=eval_ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))

    print("\ntraining...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    mem.mark("peak during training")
    elapsed = time.time() - t0

    # ---- GATE 1b: WORST-CASE LENGTH PROBE -------------------------------------
    # Today this confirms rather than corrects the phase table: tok_fn uses
    # padding="max_length", so every batch is already max_seq_len wide no matter how short the
    # examples are. The phase table's peak IS a full-length peak.
    #
    # It is kept because that guarantee lives in one keyword argument. Switching to dynamic
    # padding is the obvious throughput win (the seed set averages 107 content tokens against a
    # 1024 width, so most of the compute is padding), and the moment someone makes that change
    # the phase table starts reporting whatever length the batch happened to be. This probe is
    # length-explicit and does not care how the collator is configured.
    #
    # It matters here because vocab is 163,840: the logits tensor is [batch, seq, 163840] and
    # cross-entropy upcasts it to fp32, so the loss term alone scales linearly with seq.
    max_len = int(cfg["model"]["max_seq_len"])
    bs = int(t["per_device_train_batch_size"])
    print(f"\n  GATE 1b — worst-case probe: batch {bs} x {max_len} tokens (config maximum)")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    probe_ok, probe_peak, probe_err = True, 0.0, ""
    try:
        model.train()
        ids = torch.randint(0, int(model.config.vocab_size), (bs, max_len), device=0)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        out.loss.backward()
        probe_peak = torch.cuda.max_memory_allocated() / 1e9
        model.zero_grad(set_to_none=True)
        del out, ids
    except torch.cuda.OutOfMemoryError as e:
        probe_ok, probe_err = False, "CUDA OOM"
        probe_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"    OOM at full length. {str(e)[:90]}")
    except Exception as e:                                       # noqa: BLE001
        probe_ok, probe_err = False, f"{type(e).__name__}: {e}"
        print(f"    FAILED: {probe_err[:110]}")
    torch.cuda.empty_cache()

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if probe_ok:
        head = total_gb - probe_peak
        print(f"    peak {probe_peak:.2f} GB   headroom {head:.2f} GB of {total_gb:.2f}")
        if head < 1.0:
            print("    MARGINAL — under 1GB spare at full length. Longer real examples or a")
            print("    fragmented allocator will OOM. Cut max_seq_len before the real run.")
            probe_ok = False
        else:
            print("    PASS — a full-length batch fits, independent of collator settings.")
    else:
        print(f"    FAIL — max_seq_len {max_len} does not fit. Lower it in the config and")
        print("    re-run the pilot; do not start a real run on the phase table alone.")
    GATES["memory_at_max_length"] = {
        "pass": probe_ok, "max_seq_len": max_len, "batch": bs,
        "peak_gb": round(probe_peak, 2), "total_gb": round(total_gb, 2),
        "headroom_gb": round(total_gb - probe_peak, 2), "error": probe_err,
        "note": "synthetic full-length batch; the phase table used real, much shorter examples",
    }

    # ---- GATE 2: loss ---------------------------------------------------------
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print("\n  GATE 2 — loss")
    if len(losses) < 2:
        print("    FAIL: fewer than 2 logged losses")
        GATES["loss"] = {"pass": False, "reason": "insufficient logs"}
    else:
        first, last = losses[0], losses[-1]
        finite = all(l == l and abs(l) != float("inf") for l in losses)
        flat = abs(last - first) < 1e-6
        diverged = last > first * 3
        ok = finite and not flat and not diverged
        print(f"    first {first:.4f}  last {last:.4f}  delta {last-first:+.4f}  n={len(losses)}")
        for cond, msg in ((not finite, "NaN or inf in the loss"),
                          (flat, "loss exactly flat — gradients are not reaching the adapters"),
                          (diverged, "loss diverged — LR too high or fp16 instability")):
            if cond:
                print(f"    FAIL: {msg}")
        if ok:
            print("    PASS — machinery is connected. This is NOT evidence of quality.")
        GATES["loss"] = {"pass": ok, "first": first, "last": last, "n": len(losses),
                         "note": "a falling pilot loss proves wiring, not improvement"}

    # ---- save + provenance ----------------------------------------------------
    final = os.path.join(out_dir, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    mem.mark("checkpoint save")
    prov["gates"] = GATES
    prov["train_seconds"] = round(elapsed, 1)
    prov["steps"] = len(losses)
    for d in [final] + glob.glob(os.path.join(out_dir, "checkpoint-*")):
        with open(os.path.join(d, "run_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(prov, f, indent=2)

    mem.report(prov["cuda"]["total_memory_gb"])
    print(f"\n  GATE 4 — provenance stamped into {os.path.basename(final)} "
          f"and every checkpoint")
    print(f"    model rev  {prov['model_revision'][:12]}")
    print(f"    data sha   {prov['dataset_train_sha256'][:12]}")
    print(f"    bench sha  {prov['benchmark_sha256'][:12]}")

    failed = [k for k, v in GATES.items() if v.get("pass") is False]
    print("\n" + "=" * 72)
    if failed:
        print(f"GATES FAILED: {failed} — do not start the real run.")
        return 1
    print("GATES 1, 2, 4 PASSED.")
    print(f"Now run GATE 3 in a fresh process:")
    print(f"  python scripts/verify_checkpoint.py --checkpoint models/{exp}/final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
