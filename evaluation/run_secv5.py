"""run_secv5.py — the vuln-research DISCIPLINE benchmark (deterministic-primary).

    python evaluation/run_secv5.py --model moonlight
    python evaluation/run_secv5.py --model qwen
    python evaluation/run_secv5.py --compare moonlight qwen25-coder-14b     # CPU head-to-head
    python evaluation/run_secv5.py --report evaluation/results/secv5_moonlight/results.json

40 held-out items measuring the research-agent discipline (hypothesis-not-confirmed, validate-before-
exploit, overclaim/severity resistance, information-gain tool selection, injection-in-evidence
resistance, exploitability reasoning, taint reasoning, remediation quality). The HEADLINE is the
DETERMINISTIC overall — unbiased; the self-judge (--judge) is optional and never decides. Same
model-swap seam, greedy generation, and trust boundary as everything else.
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
sys.path.insert(0, os.path.join(HERE, "evaluation", "development", "security_v5"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
RESULTS = os.path.join(HERE, "evaluation", "results")
CAVEAT = ("HEADLINE = deterministic (unbiased, model-free). --judge is the SAME model grading itself "
          "— optimistic, a LOWER BOUND, never the verdict.")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def build_report(data: dict) -> str:
    rows = data["results"]
    det = [r["det_score"] for r in rows if r["det_score"] is not None]
    per = defaultdict(list)
    for r in rows:
        if r["det_score"] is not None:
            per[r["category"]].append(r["det_score"])
    L, A = [], lambda s: L.append(s)
    model = data.get("model", "model").split("/")[-1]
    A(f"# Security Benchmark v5 (research discipline, {len(rows)} items) — {model}\n")
    A(f"**Deterministic overall (HEADLINE, unbiased): {_mean(det)}**   over {len(det)} scored\n")
    if data.get("judge"):
        A(f"self-judge overall (optimistic LOWER BOUND): {_mean([r.get('judge_score') for r in rows])}\n")
    A("> " + CAVEAT + "\n")
    A("| Discipline category | deterministic | items |")
    A("|---|--:|--:|")
    for cat in sorted(per):
        A(f"| {cat} | {_mean(per[cat])} | {len(per[cat])} |")
    A("\n## Weakest disciplines (deterministic)")
    for cat, sc in sorted(((c, _mean(v)) for c, v in per.items()), key=lambda t: t[1])[:5]:
        A(f"- {cat}: {sc}")
    return "\n".join(L) + "\n"


def compare(a_tag: str, b_tag: str) -> str:
    def load(t):
        p = os.path.join(RESULTS, f"secv5_{t}", "results.json")
        if not os.path.exists(p):
            sys.exit(f"missing {p} — run --model {t} first")
        return json.load(open(p, encoding="utf-8"))
    A_, B_ = load(a_tag), load(b_tag)
    pa, pb = defaultdict(list), defaultdict(list)
    for r in A_["results"]:
        if r["det_score"] is not None:
            pa[r["category"]].append(r["det_score"])
    for r in B_["results"]:
        if r["det_score"] is not None:
            pb[r["category"]].append(r["det_score"])
    an = A_.get("model", a_tag).split("/")[-1]
    bn = B_.get("model", b_tag).split("/")[-1]
    L, A = [], lambda s: L.append(s)
    A("=" * 80)
    A(f"SECURITY BENCHMARK v5 — RESEARCH DISCIPLINE HEAD-TO-HEAD (deterministic) — {len(A_['results'])} items")
    A("=" * 80)
    A(f"{'discipline':28} {an[:22]:>22} {bn[:22]:>22}")
    A("-" * 80)
    for cat in sorted(set(pa) | set(pb)):
        A(f"{cat:28} {str(_mean(pa.get(cat, []))):>22} {str(_mean(pb.get(cat, []))):>22}")
    A("-" * 80)
    oa = _mean([r["det_score"] for r in A_["results"]])
    ob = _mean([r["det_score"] for r in B_["results"]])
    A(f"{'OVERALL (deterministic)':28} {str(oa):>22} {str(ob):>22}")
    gap = round((oa or 0) - (ob or 0), 3)
    if abs(gap) < 0.03:
        A(f"\nVERDICT: statistical TIE (|gap|={abs(gap)} < 0.03 over {len(A_['results'])} items).")
    else:
        A(f"\nVERDICT: {(an if gap > 0 else bn)} leads by {abs(gap)} on research discipline.")
    A("Decision rule: a material, consistent discipline edge — not speed — informs the Phase-12 pick.")
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
    ap.add_argument("--model")
    ap.add_argument("--judge", action="store_true", help="also run the optimistic self-judge (2x gens)")
    ap.add_argument("--report", help="rebuild report from a results.json (CPU)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
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

    from build_secv5 import items_as_dicts, grade_deterministic, grade_secv5
    from verification.verify import verify
    items = items_as_dicts()
    print("=" * 80)
    print(f"SECURITY BENCHMARK v5 (RESEARCH DISCIPLINE) — {len(items)} items — {CAVEAT}")
    print("=" * 80)

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
            jscore, jdetail = grade_secv5(item, answer, lambda p: gen(p, args.judge_tokens)[0])
        report = verify(answer, hits=None, tools_ran=None)
        results.append({"id": item["id"], "domain": item["domain"], "category": item["category"],
                        "det_score": det, "judge_score": jscore, "judge_detail": jdetail,
                        "verify_verdict": report.verdict, "output": answer})
        if n % 10 == 0 or n == len(items):
            print(f"  [{n}/{len(items)}] running det_overall={_mean([r['det_score'] for r in results])}")

    peak = torch.cuda.max_memory_allocated() / 1e9
    mtag = _mtag(lock)
    if args.adapter:                                  # distinct dir so base vs fine-tune --compare
        mtag += "_" + os.path.basename(os.path.dirname(args.adapter.rstrip("/\\")))
    data = {"name": f"secv5_{mtag}", "model": lock["model"], "model_revision": lock["revision"],
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
