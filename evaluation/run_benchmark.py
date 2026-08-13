"""run_benchmark.py — run the frozen benchmark against ONE model and save raw, graded results.

    python evaluation/run_benchmark.py --base
    python evaluation/run_benchmark.py --adapter models/experiment-004/final

Produces evaluation/results/<name>/results.json. Feed two of those to compare.py.

WHY BASE AND CANDIDATE ARE SEPARATE RUNS
Loading this model takes ~8 minutes. The base model's scores depend only on (revision, decode
settings, benchmark hash), so once measured they are reusable across every future experiment.
Re-running it each time would waste a quarter of a Kaggle session to reproduce a constant.

The cost of separate runs is that the two halves could drift apart. Everything that could
differ is therefore recorded in the results file, and compare.py refuses to compare two runs
whose conditions do not match. Drift becomes an error instead of a silently wrong delta.

DECODING IS GREEDY AND FIXED
do_sample=False. Not because greedy is better, but because a benchmark that samples cannot
attribute a score change to the model rather than the seed. temperature/top_p are recorded for
provenance; they have no effect while sampling is off.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# Same single-GPU pin as training: a 4-bit model on a multi-GPU box invites device confusion,
# and evaluation must run under the same conditions as training to be comparable.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

BENCH_DIR = os.path.join(HERE, "evaluation", "frozen", "benchmark_v1")

# Every knob that could change an output. Written into results.json and enforced by compare.py.
DECODE = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 512,
    "repetition_penalty": 1.0,
    "system_prompt": None,          # none, deliberately: a system prompt is a confound
    "chat_template": "model default via apply_chat_template",
}


def load_benchmark():
    items = [json.loads(l) for l in
             open(os.path.join(BENCH_DIR, "benchmark.jsonl"), encoding="utf-8")]
    lock = json.load(open(os.path.join(BENCH_DIR, "benchmark.lock.json"), encoding="utf-8"))
    return items, lock


def build_prompt(item, tok, max_ctx):
    """Assemble the user turn, attaching context for RAG items."""
    prompt = item["prompt"]
    if item.get("needs_context") and item.get("context_file"):
        ctx = open(os.path.join(BENCH_DIR, item["context_file"]), encoding="utf-8").read()
        prompt = f"{ctx}\n\n---\n\n{prompt}"
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    truncated = False
    if ids.shape[-1] > max_ctx:
        ids = ids[:, -max_ctx:]          # keep the question, which sits at the end
        truncated = True
    return ids, truncated


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", action="store_true", help="the untouched base model")
    g.add_argument("--adapter", help="path to an adapter directory")
    ap.add_argument("--out", default=None, help="results dir; derived from the model if unset")
    ap.add_argument("--limit", type=int, default=0, help="first N items (smoke testing only)")
    ap.add_argument("--max-new-tokens", type=int, default=DECODE["max_new_tokens"])
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from evaluation.grading import grade
    from training.patches import apply_all as apply_patches

    if int(transformers.__version__.split(".")[0]) >= 5:
        print(f"FAIL: transformers {transformers.__version__} cannot quantise this model's")
        print("      experts. See scripts/check_quantizable.py. Install 4.57.6.")
        return 1
    patches = apply_patches(verbose=False)

    lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))
    items, block = load_benchmark()
    if args.limit:
        items = items[:args.limit]

    name = args.out or ("base" if args.base else
                        os.path.basename(os.path.dirname(args.adapter.rstrip("/\\")))
                        or "candidate")
    out_dir = os.path.join(HERE, "evaluation", "results", name)
    os.makedirs(out_dir, exist_ok=True)

    DECODE["max_new_tokens"] = args.max_new_tokens

    print("=" * 74)
    print(f"BENCHMARK RUN — {'BASE' if args.base else args.adapter}")
    print(f"{len(items)} items   benchmark {block['sha256'][:16]}")
    print("=" * 74)

    if not torch.cuda.is_available():
        print("\nNO CUDA — this needs a GPU. Nothing to do.")
        return 2

    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    print("\nloading base weights (4-bit)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
    load_s = time.time() - t0

    if args.adapter:
        from peft import PeftModel
        ad = args.adapter if os.path.isabs(args.adapter) else os.path.join(HERE, args.adapter)
        model = PeftModel.from_pretrained(model, ad)
        acfg = json.load(open(os.path.join(ad, "adapter_config.json"), encoding="utf-8"))
        print(f"adapter applied: {sorted(acfg['target_modules'])}  r={acfg.get('r')}")
    model.eval()

    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_ctx = int(lock["context_length"]) - DECODE["max_new_tokens"]

    torch.cuda.reset_peak_memory_stats()
    results, t_start = [], time.time()

    for n, item in enumerate(items, 1):
        ids, truncated = build_prompt(item, tok, max_ctx)
        ids = ids.to(0)
        g0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=DECODE["max_new_tokens"],
                                 do_sample=DECODE["do_sample"],
                                 repetition_penalty=DECODE["repetition_penalty"],
                                 pad_token_id=tok.pad_token_id)
        latency = time.time() - g0
        new = out[0][ids.shape[-1]:]
        text = tok.decode(new, skip_special_tokens=True).strip()
        n_out = int(new.shape[-1])

        score, tier, why = grade(item, text)
        results.append({
            "id": item["id"], "category": item["category"], "layer": item["layer"],
            "grading_type": item["grading_type"], "tier": tier,
            "score": score, "explanation": why, "output": text,
            "prompt_tokens": int(ids.shape[-1]), "output_tokens": n_out,
            "latency_s": round(latency, 2),
            "tokens_per_s": round(n_out / latency, 2) if latency > 0 else 0.0,
            "context_truncated": truncated,
        })
        mark = "·" if score is None else ("+" if score >= 0.999 else
                                          ("~" if score > 0 else "-"))
        print(f"  [{n:3}/{len(items)}] {mark} {item['id']:22} "
              f"{('unscored' if score is None else f'{score:.2f}'):>8}  "
              f"{n_out:>4}tok {latency:5.1f}s")

    wall = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 1e9

    scored = [r for r in results if r["score"] is not None]
    obj = [r for r in scored if r["grading_type"] in ("exact", "code_test", "structural")]
    rub = [r for r in scored if r["grading_type"] == "rubric"]

    payload = {
        "name": name,
        "model": lock["model"],
        "model_revision": lock["revision"],
        "adapter": args.adapter,
        "adapter_config": (json.load(open(os.path.join(
            args.adapter if os.path.isabs(args.adapter) else os.path.join(HERE, args.adapter),
            "adapter_config.json"), encoding="utf-8")) if args.adapter else None),
        "benchmark_sha256": block["sha256"],
        "benchmark_items": len(items),
        "decode": DECODE,
        "upstream_patches": patches,
        "environment": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
        },
        "cost": {
            "load_seconds": round(load_s, 1),
            "wall_seconds": round(wall, 1),
            "peak_vram_gb": round(peak, 2),
            "mean_latency_s": round(sum(r["latency_s"] for r in results) / max(len(results), 1), 2),
            "mean_tokens_per_s": round(
                sum(r["tokens_per_s"] for r in results) / max(len(results), 1), 2),
            "total_output_tokens": sum(r["output_tokens"] for r in results),
        },
        "summary": {
            "objective_n": len(obj),
            "objective_mean": round(sum(r["score"] for r in obj) / len(obj), 4) if obj else None,
            "rubric_n": len(rub),
            "rubric_mean": round(sum(r["score"] for r in rub) / len(rub), 4) if rub else None,
            "unscored_n": len(results) - len(scored),
        },
        "results": results,
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    s = payload["summary"]
    print("\n" + "=" * 74)
    print(f"  objective  {s['objective_n']:>3} items   "
          f"mean {s['objective_mean'] if s['objective_mean'] is not None else '—'}")
    print(f"  rubric     {s['rubric_n']:>3} items   "
          f"mean {s['rubric_mean'] if s['rubric_mean'] is not None else '—'}")
    print(f"  unscored   {s['unscored_n']:>3} items   (judge-required or non-mechanical)")
    print(f"  cost       {wall/60:.1f} min   peak {peak:.2f} GB   "
          f"{payload['cost']['mean_tokens_per_s']} tok/s")
    print(f"\nwritten: evaluation/results/{name}/results.json")
    print("compare with:  python evaluation/compare.py --base evaluation/results/base \\")
    print(f"                                  --candidate evaluation/results/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
