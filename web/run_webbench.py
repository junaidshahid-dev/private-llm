"""run_webbench.py — run research mode over Web Benchmark v1 with the real model (Kaggle).

    python web/run_webbench.py                    # base model
    python web/run_webbench.py --model qwen       # a swapped model, same benchmark

Each item carries its own controlled web (mock search + pages), so the ONLY variable is the model's
research behaviour. Grading is deterministic (web/benchmark.grade_web): valid citations, factual
support, prompt-injection resistance, insufficiency honesty, contradiction flagging. No LLM judge,
so the score is trustworthy. Reuses the model-swap seam (--model / MODEL_LOCK).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model alias/lock (default: frozen baseline)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    from web.benchmark import ITEMS, item_searcher, item_extractor, grade_web
    from web.research import research
    from serving.model_spec import load_lock
    lock = load_lock(args.model)

    print("=" * 74)
    print(f"WEB BENCHMARK v1 — {lock['model'].split('/')[-1]} — {len(ITEMS)} items")
    print("=" * 74)

    import torch
    if not torch.cuda.is_available():
        print("NO GPU — needs the model. Run on Kaggle.")
        return 2
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    apply_all(verbose=False)
    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    print(f"loading {lock['model']} (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
    model.eval()
    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def generate(messages) -> str:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt").to(0)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        return tok.decode(o[0][ids.shape[-1]:], skip_special_tokens=True).strip()

    cfg = {"web": {"enabled": True, "search": True, "fetch": True, "private_networks": False}}
    rows, per_check = [], defaultdict(list)
    t0 = time.time()
    for n, item in enumerate(ITEMS, 1):
        rec = research(item["question"], generate, cfg,
                       searcher=item_searcher(item), extractor=item_extractor(item), k_sources=3)
        score, checks = grade_web(item, rec)
        for k, v in checks.items():
            per_check[k].append(float(v))
        rows.append({"id": item["id"], "category": item["category"], "score": score,
                     "checks": checks, "sufficient": rec["sufficient"],
                     "verify": rec["verification"]["verdict"], "answer": rec["answer"]})
        print(f"  [{n}/{len(ITEMS)}] {item['category']:20} {score:.2f}  {checks}")

    overall = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
    print("\n" + "-" * 74)
    print(f"WEB BENCHMARK overall: {overall:.3f}")
    for k in sorted(per_check):
        print(f"  {k:20} {sum(per_check[k])/len(per_check[k]):.2f}  (n={len(per_check[k])})")
    data = {"model": lock["model"], "items": len(ITEMS), "overall": round(overall, 3),
            "cost": {"seconds": round(time.time() - t0, 1)}, "results": rows}
    d = os.path.join(HERE, "evaluation", "results",
                     f"webbench_{lock['model'].split('/')[-1]}")
    os.makedirs(d, exist_ok=True)
    json.dump(data, open(os.path.join(d, "results.json"), "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> evaluation/results/{os.path.basename(d)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
