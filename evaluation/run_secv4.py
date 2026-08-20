"""run_secv4.py — the LARGER held-out security benchmark: the run that actually decides Phase 12.

    # on Kaggle (GPU), one model in VRAM at a time:
    python evaluation/run_secv4.py --model moonlight
    python evaluation/run_secv4.py --model qwen
    python evaluation/run_secv4.py --compare moonlight qwen25-coder-14b     # CPU: head-to-head
    python evaluation/run_secv4.py --report evaluation/results/secv4_qwen25-coder-14b/results.json

68 held-out items across 16 categories. The HEADLINE is the DETERMINISTIC overall — model-free and
unbiased (the self-judge has home-field bias; we do not let it decide). Over 68 items the rough
per-item anchor noise averages out, so a real quality gap shows and a tie is a tie. The optimistic
self-judge is only recorded with --judge, clearly flagged. Verification runs on every answer.

Same model-swap seam (--model / MODEL_LOCK), same greedy generation, same universal trust boundary
as everything else — only the model varies.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
V4_DIR = os.path.join(HERE, "evaluation", "development", "security_v4")
sys.path.insert(0, V4_DIR)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
RESULTS = os.path.join(HERE, "evaluation", "results")
CAVEAT = ("HEADLINE = deterministic (unbiased, model-free). The self-judge, if run, is the SAME model "
          "grading itself — optimistic, a LOWER BOUND, never the verdict.")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def build_report(data: dict) -> str:
    rows = data["results"]
    det = [r["det_score"] for r in rows if r["det_score"] is not None]
    overall_det = _mean(det)
    per = defaultdict(list)
    for r in rows:
        if r["det_score"] is not None:
            per[r["category"]].append(r["det_score"])
    L, A = [], lambda s: L.append(s)
    model = data.get("model", "model").split("/")[-1]
    A(f"# Security Benchmark v4 (held-out, {len(rows)} items) — {model}\n")
    A(f"**Deterministic overall (HEADLINE, unbiased): {overall_det}**   over {len(det)} scored\n")
    if data.get("judge"):
        A(f"self-judge overall (optimistic LOWER BOUND): {_mean([r.get('judge_score') for r in rows])}\n")
    A("> " + CAVEAT + "\n")
    A("| Category | deterministic | items |")
    A("|---|--:|--:|")
    for cat in sorted(per):
        A(f"| {cat} | {_mean(per[cat])} | {len(per[cat])} |")
    blocks = [r for r in rows if r["verify_verdict"] == "BLOCK"]
    A(f"\nVerification: {sum(1 for r in rows if r['verify_verdict']=='PASS')} PASS / "
      f"{sum(1 for r in rows if r['verify_verdict']=='WARNING')} WARN / {len(blocks)} BLOCK\n")
    A("## Weakest categories (deterministic)")
    for cat, sc in sorted(((c, _mean(v)) for c, v in per.items()), key=lambda t: t[1])[:5]:
        A(f"- {cat}: {sc}")
    return "\n".join(L) + "\n"


def compare(a_tag: str, b_tag: str) -> str:
    def load(t):
        p = os.path.join(RESULTS, f"secv4_{t}", "results.json")
        if not os.path.exists(p):
            sys.exit(f"missing {p} — run --model {t} first")
        return json.load(open(p, encoding="utf-8"))
    A_, B_ = load(a_tag), load(b_tag)
    pa = defaultdict(list)
    pb = defaultdict(list)
    for r in A_["results"]:
        if r["det_score"] is not None:
            pa[r["category"]].append(r["det_score"])
    for r in B_["results"]:
        if r["det_score"] is not None:
            pb[r["category"]].append(r["det_score"])
    an = A_.get("model", a_tag).split("/")[-1]
    bn = B_.get("model", b_tag).split("/")[-1]
    L, A = [], lambda s: L.append(s)
    A("=" * 78)
    A(f"SECURITY BENCHMARK v4 — HEAD-TO-HEAD (deterministic, unbiased) — {len(A_['results'])} items")
    A("=" * 78)
    A(f"{'category':24} {an[:22]:>24} {bn[:22]:>24}")
    A("-" * 78)
    for cat in sorted(set(pa) | set(pb)):
        A(f"{cat:24} {str(_mean(pa.get(cat, []))):>24} {str(_mean(pb.get(cat, []))):>24}")
    A("-" * 78)
    oa = _mean([r["det_score"] for r in A_["results"]])
    ob = _mean([r["det_score"] for r in B_["results"]])
    A(f"{'OVERALL (deterministic)':24} {str(oa):>24} {str(ob):>24}")
    gap = round((oa or 0) - (ob or 0), 3)
    lead = an if gap > 0 else bn
    A("")
    if abs(gap) < 0.03:
        A(f"VERDICT: statistical TIE (|gap|={abs(gap)} < 0.03 over {len(A_['results'])} items).")
    else:
        A(f"VERDICT: {lead} leads by {abs(gap)} (deterministic, unbiased).")
    A("Decision rule (his): do NOT switch on speed. A material, consistent deterministic edge across")
    A("categories — plus no tool-selection/methodology regression — is what moves Phase 12.")
    return "\n".join(L)


def _mtag(lock) -> str:
    ln = os.path.basename(lock.get("_lock_path", "MODEL_SPEC.lock.json"))
    if ln.endswith(".lock.json"):
        ln = ln[:-len(".lock.json")]
    if ln.startswith("MODEL_SPEC"):
        ln = ln[len("MODEL_SPEC"):]
    return ln.lstrip(".") or "moonlight"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="alias/lock (moonlight, qwen, qwen-14b)")
    ap.add_argument("--judge", action="store_true", help="also run the optimistic self-judge (2x gens)")
    ap.add_argument("--report", help="rebuild report from a results.json (CPU)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="CPU head-to-head of two saved runs")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--judge-tokens", type=int, default=256)
    ap.add_argument("--adapter", help="apply a trained LoRA adapter dir on top of the base "
                    "(e.g. models/experiment-001/final) to score the fine-tune vs the base")
    args = ap.parse_args()

    if args.report:
        print(build_report(json.load(open(args.report, encoding="utf-8"))))
        return 0
    if args.compare:
        print(compare(*args.compare))
        return 0

    from build_secv4 import items_as_dicts, grade_deterministic, grade_secv4
    from verification.verify import verify
    items = items_as_dicts()

    print("=" * 78)
    print(f"SECURITY BENCHMARK v4 (HELD-OUT) — {len(items)} items — {CAVEAT}")
    print("=" * 78)

    import torch
    if not torch.cuda.is_available():
        print("NO GPU — run on Kaggle. (--compare / --report are CPU-only.)")
        return 2
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    if int(transformers.__version__.split(".")[0]) >= 5:
        sys.exit("transformers 5.x cannot quantise this model; install 4.57.6.")
    apply_all(verbose=False)
    from serving.model_spec import load_lock
    lock = load_lock(args.model)
    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    print(f"loading {lock['model']} (4-bit)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
    if args.adapter:
        from peft import PeftModel
        ad = args.adapter if os.path.isabs(args.adapter) else os.path.join(HERE, args.adapter)
        model = PeftModel.from_pretrained(model, ad)
        acfg = json.load(open(os.path.join(ad, "adapter_config.json"), encoding="utf-8"))
        print(f"adapter applied: {sorted(acfg['target_modules'])}  r={acfg.get('r')}")
    load_s = time.time() - t0
    model.eval()
    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def gen(prompt, max_new):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt").to(0)
        g0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = out[0][ids.shape[-1]:]
        return tok.decode(new, skip_special_tokens=True).strip(), time.time() - g0

    torch.cuda.reset_peak_memory_stats()
    results, lat = [], 0.0
    for n, item in enumerate(items, 1):
        answer, a_lat = gen(item["prompt"], args.max_new_tokens)
        lat += a_lat
        det = grade_deterministic(item, answer)
        jscore, jdetail = (None, "")
        if args.judge:
            jscore, jdetail = grade_secv4(item, answer, lambda p: gen(p, args.judge_tokens)[0])
        report = verify(answer, hits=None, tools_ran=None)
        results.append({"id": item["id"], "domain": item["domain"], "category": item["category"],
                        "det_score": det, "judge_score": jscore, "judge_detail": jdetail,
                        "verify_verdict": report.verdict, "output": answer})
        if n % 10 == 0 or n == len(items):
            print(f"  [{n}/{len(items)}] running det_overall="
                  f"{_mean([r['det_score'] for r in results])}")

    peak = torch.cuda.max_memory_allocated() / 1e9
    mtag = _mtag(lock)
    if args.adapter:                                  # distinct dir so base vs fine-tune --compare
        mtag += "_" + os.path.basename(os.path.dirname(args.adapter.rstrip("/\\")))
    data = {"name": f"secv4_{mtag}", "model": lock["model"], "model_revision": lock["revision"],
            "adapter": args.adapter or None,
            "items": len(items), "judge": bool(args.judge),
            "environment": {"transformers": transformers.__version__,
                            "device": torch.cuda.get_device_name(0),
                            "python": platform.python_version()},
            "cost": {"load_seconds": round(load_s, 1), "peak_vram_gb": round(peak, 2),
                     "mean_latency_s": round(lat / len(items), 2)},
            "results": results}
    d = os.path.join(RESULTS, data["name"])
    os.makedirs(d, exist_ok=True)
    json.dump(data, open(os.path.join(d, "results.json"), "w", encoding="utf-8"), indent=2)
    report_md = build_report(data)
    open(os.path.join(d, "REPORT.md"), "w", encoding="utf-8").write(report_md)
    print("\n" + report_md)
    print(f"saved: evaluation/results/{data['name']}/   (run the other model, then --compare)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
