"""run_secv2.py — measure security reasoning on Benchmark v2 (base, or base + Security RAG).

    python evaluation/run_secv2.py                       # base Moonlight
    python evaluation/run_secv2.py --rag rag/sec_index   # base + Security RAG (the comparison)
    python evaluation/run_secv2.py --report <results.json>  # rebuild report only (CPU)

v2 uses the component-rubric grader (multiple required components, harmful-hallucination penalty)
so the score reflects completeness and correctness, not keyword presence. This is the instrument
for the experiment the operator wants:

    Security v2  ->  Base Moonlight  ->  Base + Security RAG  ->  compare

Run it once with no --rag (base) and once with --rag pointing at a security knowledge index; the
delta is how much the knowledge base improves reasoning without changing the model.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SECV2 = os.path.join(HERE, "evaluation", "development", "security_v2", "secv2.jsonl")


def load_items():
    if not os.path.exists(SECV2):
        sys.exit("secv2.jsonl missing — run build_secv2.py first")
    return [json.loads(l) for l in open(SECV2, encoding="utf-8") if l.strip()]


def build_report(data: dict) -> str:
    from collections import defaultdict
    rows = data["results"]
    overall = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
    harmful = [r for r in rows if r.get("harmful")]
    per = defaultdict(list)
    for r in rows:
        per[r["domain"]].append(r)
    L, A = [], None
    A = L.append
    tag = "Base + Security RAG" if data.get("rag") else "Base Moonlight"
    A(f"# Security Benchmark v2 — {tag}\n")
    A(f"{data['model']} @ {data['model_revision'][:12]}. {len(rows)} hard items, component-rubric "
      f"grading (completeness + harmful-hallucination penalty).\n")
    A(f"**Overall v2 score: {overall:.3f}**   (harmful-hallucination flags: {len(harmful)})\n")
    A("| Domain | Score | components |")
    A("|---|--:|---|")
    for dom in sorted(per):
        rs = per[dom]
        m = sum(r["score"] for r in rs) / len(rs)
        comp = "; ".join(r["explanation"].split(" required")[0] for r in rs)
        A(f"| {dom} | {m:.2f} | {comp} |")
    A(f"\n## Cost")
    c = data["cost"]
    A(f"- avg output: {sum(r['output_tokens'] for r in rows)/len(rows):.0f} tokens; "
      f"latency {c['mean_latency_s']} s/item; {c['mean_tokens_per_s']} tok/s; "
      f"peak {c['peak_vram_gb']} GB")
    if harmful:
        A(f"\n## Harmful hallucinations (confident-wrong, capped)")
        for r in harmful:
            A(f"- **{r['domain']}** ({r['id']}): {r['explanation']}")
    weak = sorted(rows, key=lambda r: r["score"])[:3]
    A(f"\n## Weakest answers (headroom for RAG)")
    for r in weak:
        A(f"- **{r['domain']}** {r['score']:.2f}: {r['output'][:150].strip()}…")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", help="security index dir to retrieve context from")
    ap.add_argument("--report", help="rebuild report from an existing results.json (CPU)")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    if args.report:
        print(build_report(json.load(open(args.report, encoding="utf-8"))))
        return 0

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    sys.path.insert(0, os.path.dirname(SECV2))
    from build_secv2 import grade_secv2

    if int(transformers.__version__.split(".")[0]) >= 5:
        sys.exit("transformers 5.x cannot quantise this model; install 4.57.6.")
    apply_all(verbose=False)
    lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))
    items = load_items()

    retriever = None
    if args.rag:
        from rag.store import Store
        from rag.query import build_prompt, MIN_SCORE
        retriever = (Store().load(args.rag if os.path.isabs(args.rag)
                                  else os.path.join(HERE, args.rag)), build_prompt, MIN_SCORE)

    mode = "BASE + RAG" if args.rag else "BASE"
    print("=" * 74)
    print(f"SECURITY BENCHMARK v2 — {mode} — {len(items)} hard items")
    print("=" * 74)
    if not torch.cuda.is_available():
        print("NO GPU — run on Kaggle. (--report <results.json> rebuilds the report on CPU.)")
        return 2

    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    print("loading base Moonlight (4-bit, no adapter)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
    load_s = time.time() - t0
    model.eval()
    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.cuda.reset_peak_memory_stats()
    results, t_start = [], time.time()
    for n, item in enumerate(items, 1):
        prompt = item["prompt"]
        if retriever:
            store, build_prompt, _ = retriever
            hits = store.search(item["prompt"], k=args.k)
            prompt = build_prompt(item["prompt"], hits)
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt").to(0)
        g0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        latency = time.time() - g0
        new = out[0][ids.shape[-1]:]
        text = tok.decode(new, skip_special_tokens=True).strip()
        score, why = grade_secv2(item, text)
        results.append({"id": item["id"], "domain": item["domain"], "score": score,
                        "explanation": why, "harmful": "HARMFUL" in why, "output": text,
                        "output_tokens": int(new.shape[-1]), "latency_s": round(latency, 2),
                        "tokens_per_s": round(int(new.shape[-1]) / latency, 2) if latency else 0})
        print(f"  [{n}/{len(items)}] {item['domain']:22} {score:.2f}  {why}")

    peak = torch.cuda.max_memory_allocated() / 1e9
    data = {"name": "secv2_" + ("rag" if args.rag else "base"), "model": lock["model"],
            "model_revision": lock["revision"], "rag": bool(args.rag), "items": len(items),
            "environment": {"transformers": transformers.__version__,
                            "device": torch.cuda.get_device_name(0),
                            "python": platform.python_version()},
            "cost": {"load_seconds": round(load_s, 1),
                     "peak_vram_gb": round(peak, 2),
                     "mean_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 2),
                     "mean_tokens_per_s": round(sum(r["tokens_per_s"] for r in results) / len(results), 2)},
            "results": results}
    out_dir = os.path.join(HERE, "evaluation", "results", data["name"])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(data, open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8"), indent=2)
    report = build_report(data)
    open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8").write(report)
    print("\n" + report)
    print(f"saved: evaluation/results/{data['name']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
