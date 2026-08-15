"""run_seccap.py — Moonlight Security Baseline v1. Evaluation only, base model, no RAG.

    python evaluation/run_seccap.py               # needs the GPU (Kaggle)
    python evaluation/run_seccap.py --report path/to/results.json   # rebuild report only (CPU)

Strictly a measurement: base Moonlight, pinned model/config, the frozen security-capability items,
no adapter, no RAG, no policy prompt — so the number is the model's own raw security reasoning.
The benchmark is never used as RAG or training data (evaluation/ is skipped by both).

Produces the baseline report the operator asked for: overall + per-domain scores, average output
length, latency / tokens-per-second, peak VRAM, deterministic vs judge-recommended split, and
examples of the worst and best answers. Freeze this result as the number every future change must
beat.
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

SECCAP = os.path.join(HERE, "evaluation", "development", "security_capability", "seccap.jsonl")
OUT_DIR = os.path.join(HERE, "evaluation", "results", "seccap_base_v1")
MAX_NEW_TOKENS = 512


def load_items():
    if not os.path.exists(SECCAP):
        sys.exit("seccap.jsonl missing — run build_seccap.py first")
    return [json.loads(l) for l in open(SECCAP, encoding="utf-8") if l.strip()]


def build_report(data: dict) -> str:
    from collections import defaultdict
    rows = data["results"]
    scored = [r for r in rows if r["score"] is not None]
    overall = sum(r["score"] for r in scored) / len(scored) if scored else 0.0

    per = defaultdict(list)
    for r in rows:
        per[r["domain"]].append(r)

    det = [r for r in scored if not r.get("judge_recommended")]
    jud = [r for r in scored if r.get("judge_recommended")]
    det_mean = sum(r["score"] for r in det) / len(det) if det else None
    jud_mean = sum(r["score"] for r in jud) / len(jud) if jud else None
    avg_len = sum(r["output_tokens"] for r in rows) / len(rows) if rows else 0

    L = []
    A = L.append
    A(f"# Moonlight Security Baseline v1\n")
    A(f"Base {data['model']} @ {data['model_revision'][:12]}, evaluation only "
      f"(no adapter, no RAG, no policy prompt). {len(rows)} items.\n")
    A(f"**Overall security score: {overall:.3f}**  ({len(scored)}/{len(rows)} scored)\n")
    A("| Metric | Score | n |")
    A("|---|--:|--:|")
    for dom in sorted(per):
        rs = [r for r in per[dom] if r["score"] is not None]
        m = sum(r["score"] for r in rs) / len(rs) if rs else None
        A(f"| {dom} | {m:.2f} | {len(per[dom])} |" if m is not None else f"| {dom} | — | {len(per[dom])} |")
    A(f"\n## Reliability of the number")
    A(f"- deterministic (keyword-conclusion) items: {det_mean if det_mean is None else round(det_mean,3)}"
      f"  ({len(det)} items)")
    A(f"- judge-recommended items (deeper reasoning, scored by heuristic for now): "
      f"{jud_mean if jud_mean is None else round(jud_mean,3)}  ({len(jud)} items)")
    A(f"  These need an LLM judge for a trustworthy read; the heuristic only checks the conclusion.")
    A(f"\n## Cost")
    c = data["cost"]
    A(f"- average output length: {avg_len:.0f} tokens")
    A(f"- latency: {c['mean_latency_s']} s/item mean; {c['mean_tokens_per_s']} tok/s")
    A(f"- peak VRAM: {c['peak_vram_gb']} GB")
    A(f"- decode: greedy, max_new_tokens {data['decode']['max_new_tokens']}")

    fails = sorted([r for r in scored if r["score"] == 0], key=lambda r: r["domain"])[:4]
    wins = sorted([r for r in scored if r["score"] == 1], key=lambda r: -r["output_tokens"])[:3]
    A(f"\n## Major failures (score 0)")
    for r in fails:
        A(f"- **{r['domain']}** ({r['id']}): {r['output'][:180].strip()}…")
    A(f"\n## Strong answers (score 1)")
    for r in wins:
        A(f"- **{r['domain']}** ({r['id']}): {r['output'][:180].strip()}…")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="rebuild the report from an existing results.json (CPU)")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    if args.report:
        data = json.load(open(args.report, encoding="utf-8"))
        print(build_report(data))
        return 0

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    sys.path.insert(0, os.path.dirname(SECCAP))
    from build_seccap import grade_seccap

    if int(transformers.__version__.split(".")[0]) >= 5:
        sys.exit("transformers 5.x cannot quantise this model; install 4.57.6.")
    apply_all(verbose=False)
    lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))
    items = load_items()

    print("=" * 74)
    print(f"MOONLIGHT SECURITY BASELINE v1 — {len(items)} items, base model, eval only")
    print("=" * 74)
    if not torch.cuda.is_available():
        print("NO GPU — run this on Kaggle. (Use --report <results.json> to rebuild the report.)")
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
        ids = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}],
                                      add_generation_prompt=True, return_tensors="pt").to(0)
        g0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        latency = time.time() - g0
        new = out[0][ids.shape[-1]:]
        text = tok.decode(new, skip_special_tokens=True).strip()
        score, why = grade_seccap(item, text)
        results.append({"id": item["id"], "domain": item["domain"], "score": score,
                        "explanation": why, "output": text, "output_tokens": int(new.shape[-1]),
                        "latency_s": round(latency, 2),
                        "tokens_per_s": round(int(new.shape[-1]) / latency, 2) if latency else 0,
                        "judge_recommended": item.get("judge_recommended", False)})
        mark = "+" if score >= 0.999 else "-"
        print(f"  [{n:2}/{len(items)}] {mark} {item['domain']:26} {score:.2f}  "
              f"{int(new.shape[-1])}tok {latency:5.1f}s")

    peak = torch.cuda.max_memory_allocated() / 1e9
    data = {
        "name": "seccap_base_v1", "model": lock["model"], "model_revision": lock["revision"],
        "benchmark": "security_capability seed", "items": len(items),
        "decode": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "environment": {"transformers": transformers.__version__, "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(0), "python": platform.python_version()},
        "cost": {"load_seconds": round(load_s, 1), "wall_seconds": round(time.time() - t_start, 1),
                 "peak_vram_gb": round(peak, 2),
                 "mean_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 2),
                 "mean_tokens_per_s": round(sum(r["tokens_per_s"] for r in results) / len(results), 2)},
        "results": results}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(data, open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8"), indent=2)
    report = build_report(data)
    open(os.path.join(OUT_DIR, "REPORT.md"), "w", encoding="utf-8").write(report)
    print("\n" + report)
    print(f"saved: evaluation/results/seccap_base_v1/{{results.json, REPORT.md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
