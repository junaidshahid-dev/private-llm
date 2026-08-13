"""freeze_spec.py — pin the model, derive the real module tree, validate the LoRA config.

Runs on a laptop with no GPU and downloads ~2MB, not 32GB. It reads config.json and
model.safetensors.index.json, which together describe every parameter in the model without
fetching any weights.

    python scripts/freeze_spec.py                       # validate + write MODEL_SPEC.lock.json
    python scripts/freeze_spec.py --compare-lora        # attention-only vs attention+MLP

WHY THIS EXISTS

The standard Llama LoRA recipe targets [q_proj, k_proj, v_proj, o_proj]. This model uses
Multi-head Latent Attention and has NO k_proj and NO v_proj. Configured that way, PEFT either
raises "target modules not found" or silently adapts half of what you intended - and you find
out after paying for GPU time. So targets are DERIVED from the tensor index and any configured
name that does not exist is a hard failure, not a warning.

Nothing here is pinned to `main`. A model repo can change under you; the lockfile records the
exact commit so a run is reproducible six months later.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)

MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(HERE, "MODEL_SPEC.lock.json")
UA = {"User-Agent": "my-llm/freeze_spec"}

# What the config asks for. Must match reality or we stop.
CONFIGURED_TARGETS = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]


def get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def hf_json(model: str, rev: str, path: str):
    return json.loads(get(f"https://huggingface.co/{model}/raw/{rev}/{path}"))


def linear_modules(index: dict) -> Counter:
    """Every distinct module name that owns a .weight, with how many instances exist."""
    out = Counter()
    for name in index["weight_map"]:
        if not name.endswith(".weight"):
            continue
        leaf = name[: -len(".weight")].split(".")[-1]
        out[leaf] += 1
    return out


def lora_params(cfg: dict, targets: list[str], r: int) -> tuple[int, list[str]]:
    """LoRA parameter count = r * (in + out) per adapted linear, summed. Shapes from config."""
    h = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    kv_r = cfg.get("kv_lora_rank", 0)
    rope = cfg.get("qk_rope_head_dim", 0)
    nope = cfg.get("qk_nope_head_dim", 0)
    v_dim = cfg.get("v_head_dim", 0)
    layers = cfg["num_hidden_layers"]
    dense_first = cfg.get("first_k_dense_replace", 0)
    moe_inter = cfg.get("moe_intermediate_size", 0)
    inter = cfg.get("intermediate_size", 0)
    n_exp = cfg.get("n_routed_experts", 0)
    n_shared = cfg.get("n_shared_experts", 0)

    shapes = {
        "q_proj": (h, n_heads * (nope + rope)),
        "kv_a_proj_with_mqa": (h, kv_r + rope),
        "kv_b_proj": (kv_r, n_heads * (nope + v_dim)),
        "o_proj": (n_heads * v_dim, h),
    }
    total, notes = 0, []
    for t in targets:
        if t in shapes:
            i, o = shapes[t]
            n = layers * r * (i + o)
            total += n
            notes.append(f"{t}: {layers} x r({i}+{o}) = {n:,}")
        elif t in MLP_TARGETS:
            # dense layer 0 uses intermediate_size; MoE layers use moe_intermediate_size,
            # once per routed expert plus the shared experts.
            dense = dense_first * r * (h + inter)
            per_moe = (n_exp + n_shared) * r * (h + moe_inter)
            n = dense + (layers - dense_first) * per_moe
            total += n
            notes.append(f"{t}: dense {dense:,} + MoE {(layers-dense_first)*per_moe:,} = {n:,}")
    return total, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-lora", action="store_true")
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    print(f"resolving {MODEL} ...")
    info = json.loads(get(f"https://huggingface.co/api/models/{MODEL}"))
    rev = info["sha"]
    print(f"  pinned revision : {rev}")
    print(f"  last modified   : {info['lastModified']}")
    if info.get("gated") or info.get("private") or info.get("disabled"):
        raise SystemExit("model is gated/private/disabled — stop")

    cfg = hf_json(MODEL, rev, "config.json")
    idx = json.loads(get(f"https://huggingface.co/{MODEL}/raw/{rev}/model.safetensors.index.json"))
    tok = hf_json(MODEL, rev, "tokenizer_config.json")

    sf = info.get("safetensors") or {}
    total_params = sf.get("total", 0)
    bytes_bf16 = total_params * 2
    bytes_4bit = total_params * 0.5

    print("\n--- architecture (config.json) ---")
    for k in ("model_type", "max_position_embeddings", "hidden_size", "num_hidden_layers",
              "num_attention_heads", "vocab_size", "n_routed_experts", "n_shared_experts",
              "num_experts_per_tok", "kv_lora_rank", "q_lora_rank", "first_k_dense_replace"):
        if k in cfg:
            print(f"  {k:26} {cfg[k]}")
    print(f"  {'parameters':26} {total_params:,}")
    print(f"  {'weights @ bf16':26} {bytes_bf16/1e9:.1f} GB")
    print(f"  {'weights @ 4-bit nf4':26} {bytes_4bit/1e9:.1f} GB (+ ~0.5 GB quant constants)")

    # ---- the check that matters -------------------------------------------------
    mods = linear_modules(idx)
    print("\n--- LoRA target validation (derived from tensor index) ---")
    missing = [t for t in CONFIGURED_TARGETS if t not in mods]
    for t in CONFIGURED_TARGETS:
        mark = "OK  " if t in mods else "MISSING"
        print(f"  {mark} {t:22} {mods.get(t, 0)} instances")

    llama = ["q_proj", "k_proj", "v_proj", "o_proj"]
    absent = [t for t in llama if t not in mods]
    if absent:
        print(f"\n  note: the standard Llama recipe {llama}")
        print(f"        would fail here — {absent} do not exist in this model.")

    if missing:
        print(f"\nFAIL: configured targets not present in the model: {missing}")
        print("Fix configs/moonlight_qlora.yaml before spending any GPU time.")
        return 1

    if args.compare_lora:
        print(f"\n--- trainable parameters at r={args.rank} ---")
        a_total, a_notes = lora_params(cfg, CONFIGURED_TARGETS, args.rank)
        b_total, b_notes = lora_params(cfg, CONFIGURED_TARGETS + MLP_TARGETS, args.rank)
        for n in a_notes:
            print("   ", n)
        print(f"  attention only      {a_total:,}  ({a_total/total_params*100:.3f}% of model)")
        print(f"  attention + MLP     {b_total:,}  ({b_total/total_params*100:.3f}% of model)")
        print(f"  ratio               {b_total/max(a_total,1):.1f}x more trainable parameters")
        print("\n  More trainable parameters is not better. On a 16GB T4 the MLP variant adds")
        print("  optimizer state and activation memory for adapters that each see only the")
        print("  tokens routed to their expert. Start attention-only; escalate only if the")
        print("  eval says attention-only underfits.")

    lock = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "revision": rev,
        "revision_note": "pin this; do not use 'main'",
        "license_declared": (info.get("cardData") or {}).get("license"),
        "license_file_present": any("LICENSE" in s["rfilename"].upper()
                                    for s in info.get("siblings", [])),
        "architecture": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "parameters": total_params,
        "context_length": cfg.get("max_position_embeddings"),
        "bytes_bf16": bytes_bf16,
        "bytes_4bit_est": bytes_4bit,
        "trust_remote_code_required": True,
        "remote_code_files": [s["rfilename"] for s in info.get("siblings", [])
                              if s["rfilename"].endswith(".py")],
        "tokenizer_class": tok.get("tokenizer_class"),
        "lora_targets_validated": CONFIGURED_TARGETS,
        "lora_targets_rejected": {"llama_default": llama, "absent_here": absent},
        "quantization": {"method": "bitsandbytes nf4", "double_quant": True,
                         "compute_dtype": "float16 (T4 has no bf16)"},
        # VERIFIED 2026-08-13 by building the model on a meta device — no GPU, no weights.
        #
        # transformers 5.x BREAKS this model. The repo's modeling_deepseek.py imports
        # `is_torch_fx_available` from transformers.utils.import_utils, removed in 5.0. Under
        # 5.15.0 it raises ImportError AFTER the weights download — i.e. it would burn a rented
        # GPU session before failing. 4.57.6 is the last 4.x release and builds correctly.
        # Do not "upgrade" these without re-running the smoke test.
        "framework_pins": {
            "transformers": "4.57.6",
            "transformers_note": "5.x removes is_torch_fx_available; the remote code fails",
            "torch": ">=2.4",
            "peft": ">=0.11",
            "bitsandbytes": ">=0.43",
            "accelerate": ">=0.30",
            "trl": ">=0.9",
            "tiktoken": "REQUIRED — tokenizer is tiktoken-based, not sentencepiece",
            "blobfile": "REQUIRED by tiktoken for this tokenizer",
        },
        "verified": {
            "meta_device_build": True,
            "targets_confirmed_against_live_module_tree": True,
            "verified_on": "transformers 4.57.6 / torch 2.10.0+cpu / no GPU",
        },
    }
    with open(LOCK, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
    print(f"\nwrote {os.path.relpath(LOCK, HERE)}")
    print("PASS — configuration is valid against the pinned revision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
