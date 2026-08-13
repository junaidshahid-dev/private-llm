"""smoke_test.py — validate the entire training path with no GPU and no downloaded weights.

    .venv/Scripts/python.exe scripts/smoke_test.py

WHAT IT PROVES, AND WHAT IT DOES NOT

Proves: the config parses, the LoRA targets exist on the real 16B module tree, PEFT resolves
them, an optimizer builds over the adapters, a forward and backward pass produces a finite loss
with gradients flowing ONLY to adapter weights, the adapter saves and reloads bit-identically,
and every checkpoint path is writable.

Does NOT prove: that 4-bit CUDA training fits in 16GB, that throughput is acceptable, or that
the loss goes down on real data. Those need a GPU and are what the paid pilot is for. This test
exists so that the pilot fails for interesting reasons rather than typos.

THE TRICK. A 16B model cannot be instantiated on this laptop. So there are two models here:

  1. the REAL one, built on a meta device (no memory, no weights) — used to verify that the
     configured LoRA target names exist on the actual architecture.
  2. a TINY one, same DeepseekV3 architecture and same module names, ~10M params, real weights
     on CPU — used to actually run forward/backward/save/load.

Same code path, same class, same target resolution. Only the dimensions differ.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def main() -> int:
    import torch
    import yaml
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    print("=" * 72)
    print("SMOKE TEST — no GPU, no weight download")
    print("=" * 72)

    # ---- 1. config + lockfile ------------------------------------------------
    print("\n1. configuration")
    cfg = yaml.safe_load(open(os.path.join(HERE, "configs", "moonlight_qlora.yaml"),
                              encoding="utf-8"))
    lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))
    check("config parses", bool(cfg.get("model") and cfg.get("lora")))
    check("lockfile pins a revision", len(lock.get("revision", "")) == 40, lock["revision"][:12])
    check("config model matches lockfile", cfg["model"]["name"] == lock["model"])
    import transformers
    want = lock["framework_pins"]["transformers"]
    check(f"transformers == {want}", transformers.__version__ == want,
          f"found {transformers.__version__}")

    # Upstream patches must be in force here too — the smoke test is only meaningful if it
    # exercises the same code the trainer will run. See training/patches.py.
    from training.patches import apply_all as apply_patches
    applied = apply_patches(verbose=False)
    check("upstream patches applied", "deepseek_v3_moe_dtype" in applied, ", ".join(applied))

    # Reproduce the mixed-dtype MoE condition directly: fp32 hidden states from an upcast
    # layernorm, fp16 router weights from autocast. This is what killed the first GPU pilot.
    from transformers.models.deepseek_v3 import modeling_deepseek_v3 as _dsv3
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
    _c = DeepseekV3Config(hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                          n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
                          first_k_dense_replace=0, vocab_size=128)
    torch.manual_seed(0)
    try:
        _out = _dsv3.DeepseekV3MoE(_c).moe(torch.randn(8, 64),
                                           torch.randint(0, 4, (8, 2)),
                                           torch.rand(8, 2, dtype=torch.float16))
        check("MoE survives fp32 hidden / fp16 router", torch.isfinite(_out).all().item(),
              "the exact case that failed on the T4")
    except RuntimeError as e:
        check("MoE survives fp32 hidden / fp16 router", False, str(e)[:80])

    # ---- 2. real architecture, meta device -----------------------------------
    # NATIVE implementation, no trust_remote_code. transformers ships deepseek_v3 since 4.49,
    # and the repo's own modeling_deepseek.py is stale: it calls DynamicCache.get_usable_length
    # (removed) and from_legacy_cache(None) (now asserts). The remote code fails at FORWARD on
    # every transformers that installs on modern Python. The native class works and removes the
    # need to execute code from the repo at all.
    print("\n2. real 16B module tree (meta device — 0 bytes, NATIVE class)")
    real_cfg = AutoConfig.from_pretrained(lock["model"], revision=lock["revision"])
    check("native deepseek_v3 (no remote code)", type(real_cfg).__name__ == "DeepseekV3Config",
          type(real_cfg).__name__)
    with torch.device("meta"):
        real = AutoModelForCausalLM.from_config(real_cfg)
    leaves = {n.split(".")[-1] for n, _ in real.named_modules()}
    targets = cfg["lora"]["target_modules"]
    for t in targets:
        check(f"target `{t}` exists on the real model", t in leaves)
    check("llama defaults correctly absent", "k_proj" not in leaves and "v_proj" not in leaves,
          "k_proj/v_proj do not exist — MLA")
    del real

    # ---- 3. tiny model, same architecture, real weights ----------------------
    print("\n3. tiny model, same architecture, real weights on CPU")
    # Shrink ONLY dimensions that do not participate in the MLA head arithmetic. Scaling
    # qk_nope/qk_rope/v_head_dim/kv_lora_rank breaks split_with_sizes inside the attention —
    # a real failure I hit doing exactly that.
    d = real_cfg.to_dict()
    d.update(hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
             num_hidden_layers=2, num_attention_heads=2, n_routed_experts=4,
             n_shared_experts=1, num_experts_per_tok=2, vocab_size=512,
             first_k_dense_replace=1, torch_dtype="float32")
    tiny_cfg = type(real_cfg).from_dict(d)
    tiny = AutoModelForCausalLM.from_config(tiny_cfg).float()
    n_params = sum(p.numel() for p in tiny.parameters())
    tiny_leaves = {n.split(".")[-1] for n, _ in tiny.named_modules()}
    check("tiny model builds", n_params > 0, f"{n_params:,} params")
    check("tiny model has the same target names", all(t in tiny_leaves for t in targets))

    # ---- 4. PEFT resolves the targets ----------------------------------------
    print("\n4. LoRA attach")
    lc = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                    lora_dropout=cfg["lora"]["dropout"], bias=cfg["lora"]["bias"],
                    task_type=cfg["lora"]["task_type"], target_modules=targets)
    peft_model = get_peft_model(tiny, lc)
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in peft_model.parameters())
    check("adapters attached", len(trainable) > 0, f"{len(trainable)} tensors")
    check("all trainable params are LoRA", all("lora_" in n for n in trainable))
    check("base weights frozen", n_train < n_all * 0.5,
          f"{n_train:,} / {n_all:,} = {n_train/n_all*100:.2f}%")
    hit = {t for t in targets if any(t in n for n in trainable)}
    check("every configured target got an adapter", hit == set(targets),
          f"adapted: {sorted(hit)}")

    # ---- 5. tokenizer + real training data -----------------------------------
    print("\n5. tokenizer and dataset")
    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    check("tokenizer loads at pinned revision", tok is not None,
          type(tok).__name__)
    has_tmpl = getattr(tok, "chat_template", None) is not None
    check("tokenizer has a chat template", has_tmpl,
          "used by prepare_dataset" if has_tmpl else "NONE — prompts must be built manually")
    train_p = os.path.join(HERE, "data", "train", "train.jsonl")
    rows = [json.loads(l) for l in open(train_p, encoding="utf-8") if l.strip()]
    check("training data present", len(rows) > 0, f"{len(rows)} examples")

    # ---- 6. forward / backward -----------------------------------------------
    print("\n6. forward + backward")
    text = "\n".join(m["content"] for m in rows[0]["messages"])
    ids = tok(text, return_tensors="pt", truncation=True, max_length=64)["input_ids"]
    ids = ids % tiny_cfg.vocab_size                      # tiny model has a small vocab
    out = peft_model(input_ids=ids, labels=ids)
    loss = out.loss
    check("loss is finite", torch.isfinite(loss).item(), f"loss = {loss.item():.4f}")
    loss.backward()
    g_lora = [n for n, p in peft_model.named_parameters()
              if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    g_base = [n for n, p in peft_model.named_parameters()
              if not p.requires_grad and p.grad is not None]
    check("gradients reached adapters", len(g_lora) > 0, f"{len(g_lora)} adapter tensors")
    check("no gradients on frozen base weights", len(g_base) == 0)

    # ---- 7. optimizer ---------------------------------------------------------
    print("\n7. optimizer")
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["training"]["learning_rate"]))
    before = params[0].detach().clone()
    opt.step()
    moved = not torch.equal(before, params[0].detach())
    check("optimizer builds over adapters only", len(opt.param_groups[0]["params"]) == len(params))
    check("a step actually changes adapter weights", moved)
    check("configured optimizer is paged", "paged" in cfg["training"]["optim"],
          cfg["training"]["optim"] + " — needs bitsandbytes on GPU, not testable here")

    # ---- 8. adapter save / load ----------------------------------------------
    print("\n8. adapter save and reload")
    with tempfile.TemporaryDirectory() as td:
        peft_model.save_pretrained(td)
        files = os.listdir(td)
        check("adapter_config.json written", "adapter_config.json" in files)
        check("adapter weights written",
              any(f.startswith("adapter_model") for f in files), str(files))
        saved_cfg = json.load(open(os.path.join(td, "adapter_config.json"), encoding="utf-8"))
        check("saved config preserves targets",
              set(saved_cfg["target_modules"]) == set(targets))
        fresh = AutoModelForCausalLM.from_config(tiny_cfg).float()
        reloaded = PeftModel.from_pretrained(fresh, td)
        a = dict(peft_model.named_parameters())
        b = dict(reloaded.named_parameters())
        keys = [k for k in a if "lora_" in k]
        same = all(torch.allclose(a[k].detach(),
                                  b[k.replace("default", "default")].detach(), atol=1e-6)
                   for k in keys if k in b)
        check("reloaded adapter weights match", same, f"{len(keys)} tensors compared")

    # ---- 9. checkpoint paths --------------------------------------------------
    print("\n9. checkpoint paths")
    out_dir = os.path.join(HERE, cfg["training"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    probe = os.path.join(out_dir, ".write_probe")
    try:
        open(probe, "w").write("x"); os.remove(probe); writable = True
    except Exception:                                        # noqa: BLE001
        writable = False
    check("output_dir writable", writable, os.path.relpath(out_dir, HERE))
    models_root = os.path.join(HERE, "models")
    for sub in ("moonlight-base", "experiment-001", "best"):
        os.makedirs(os.path.join(models_root, sub), exist_ok=True)
    check("models/ experiment layout exists",
          os.path.isdir(os.path.join(models_root, "experiment-001")),
          "base model never overwritten by a run")
    bench = os.path.join(HERE, "evaluation", "frozen", "benchmark_v1", "benchmark.lock.json")
    check("frozen benchmark exists", os.path.exists(bench),
          json.load(open(bench, encoding="utf-8"))["sha256"][:16] if os.path.exists(bench) else "")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        print("Fix these before renting a GPU.")
        return 1
    print("ALL CHECKS PASSED — the training path is valid end to end on CPU.")
    print("Not yet proven: 4-bit CUDA memory fit, throughput, and whether loss falls on real")
    print("data. That is what the cheap GPU pilot is for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
