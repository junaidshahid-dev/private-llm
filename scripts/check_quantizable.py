"""check_quantizable.py — how much of the model can bitsandbytes actually quantise?

    python scripts/check_quantizable.py

This is the evidence behind the transformers version cap in MODEL_SPEC.lock.json. It downloads
nothing: the model is built on the meta device, so it costs 0 bytes and a few seconds.

WHY THIS EXISTS

bitsandbytes quantises by REPLACING nn.Linear modules with Linear4bit. Anything stored as a raw
nn.Parameter is invisible to it and stays at full precision. That is normally irrelevant — until
a library refactors how it stores weights.

transformers 5.0 did exactly that to DeepSeek-V3 MoE. Experts moved from

    self.experts = nn.ModuleList([DeepseekV3MLP(...) for _ in range(n_routed_experts)])

to 3D parameter tensors

    self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate, hidden))

which is faster and cleaner, and completely un-quantisable by bnb as it currently stands.
Measured on Moonlight-16B-A3B-Instruct:

    transformers 4.57.6    97.9% in nn.Linear     8.49 GB    fits a 15.64GB T4
    transformers 5.15.0     7.7% in nn.Linear    30.08 GB    fits nothing free

Nothing in the model is broken on 5.x — the module tree is correct and the LoRA targets still
resolve. It simply stops fitting, which on free hardware is the same as not working.

Re-run this when upgrading. If a future bitsandbytes or transformers learns to quantise 3D
expert tensors, the percentage jumps back up and the cap can be lifted.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

T4_GB = 15.64
FIT_CEILING_GB = 14.0        # leave room for activations, gradients, optimiser, fragmentation


def main() -> int:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))

    print("=" * 72)
    print(f"QUANTISABLE FRACTION — transformers {transformers.__version__}")
    print(f"{lock['model']} @ {lock['revision'][:12]}")
    print("=" * 72)

    cfg = AutoConfig.from_pretrained(lock["model"], revision=lock["revision"])
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    linear_ids, linear_params = set(), 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            for p in mod.parameters(recurse=False):
                if id(p) not in linear_ids:
                    linear_ids.add(id(p))
                    linear_params += p.numel()

    total, leftovers = 0, []
    for name, p in model.named_parameters():
        total += p.numel()
        if id(p) not in linear_ids:
            leftovers.append((name, p.numel(), tuple(p.shape)))

    nonlinear = sum(n for _, n, _ in leftovers)
    pct = 100.0 * linear_params / total
    est = (linear_params * 0.5 + nonlinear * 2) / 1e9      # 4-bit + fp16

    print(f"\n  total parameters        {total:>16,}")
    print(f"  in nn.Linear -> 4-bit    {linear_params:>16,}   {pct:5.1f}%")
    print(f"  raw Parameter -> fp16    {nonlinear:>16,}   {100-pct:5.1f}%")
    print(f"\n  weights VRAM estimate    {est:.2f} GB"
          f"   ({linear_params*0.5/1e9:.2f} 4-bit + {nonlinear*2/1e9:.2f} fp16)")

    fits = est < FIT_CEILING_GB
    print(f"  T4 ({T4_GB} GB)            {'FITS' if fits else 'DOES NOT FIT'}"
          f"   (ceiling {FIT_CEILING_GB} GB, leaving room for training state)")

    big = sorted(leftovers, key=lambda x: -x[1])[:4]
    if big and big[0][1] > 1_000_000:
        print("\n  largest tensors bnb cannot reach:")
        for n, c, s in big:
            print(f"    {c:>15,}  {n[:52]:52} {s}")
        if any("expert" in n for n, _, _ in big):
            print("\n  Experts are stored as raw parameters — this is the transformers 5.x MoE")
            print("  refactor. bitsandbytes cannot quantise them, so the model no longer fits.")
            print("  Stay on transformers 4.57.6 until bnb gains 3D expert support.")

    print("\n" + "=" * 72)
    if not fits:
        print("FAIL — this transformers version cannot run the model on free hardware.")
        return 1
    print(f"PASS — {pct:.1f}% quantisable, {est:.2f} GB of weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
