"""run_secv3.py — held-out Security Benchmark v3: base vs base+RAG, with the verification layer.

    python evaluation/run_secv3.py                     # Base Moonlight
    python evaluation/run_secv3.py --rag rag/index     # Base + Security RAG
    python evaluation/run_secv3.py --report <results.json>   # rebuild report only (CPU)

Three things at once, per the plan:
  1. CAPABILITY on HELD-OUT security reasoning, judged SEMANTICALLY (an LLM judge scores each
     rubric point present/absent), so a differently-worded correct answer wins and a shallow one
     does not. Run once without --rag and once with --rag; the delta is generalisation from
     retrieval on topics the KB does NOT contain.
  2. VERIFICATION on every answer (verification/verify.py), recording its PASS/WARNING/BLOCK
     verdict. The verifier never changes the answer or the capability score — it is measured
     separately, as a reliability layer.
  3. The comparison the operator asked for: Base -> Base+RAG -> (+Verification). Verification's
     value is not a higher capability number; it is catching mechanical defects (arithmetic,
     phantom tool claims, unsupported/mis-numbered citations) that the raw answer hides.

HONESTY: offline, the judge is Moonlight grading itself — an optimistic grader (a model rarely
catches the errors it would make). The v3 number is therefore a LOWER BOUND on rigour, not a
verdict; treat a strong result as 'worth a human spot-check', not proof. This caveat is printed
with the score, by design.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SECV3_DIR = os.path.join(HERE, "evaluation", "development", "security_v3")
SECV3 = os.path.join(SECV3_DIR, "secv3.jsonl")
CAVEAT = ("NOTE: judge = Moonlight grading itself (optimistic). This score is a LOWER BOUND on "
          "rigour, not a verdict — a strong result warrants a human spot-check, not a claim of "
          "proof.")


def load_items():
    if not os.path.exists(SECV3):
        sys.exit("secv3.jsonl missing — run build_secv3.py first")
    return [json.loads(l) for l in open(SECV3, encoding="utf-8") if l.strip()]


def build_report(data: dict) -> str:
    rows = data["results"]
    scored = [r for r in rows if r["score"] is not None]
    overall = sum(r["score"] for r in scored) / len(scored) if scored else 0.0
    unscored = [r for r in rows if r["score"] is None]
    per = defaultdict(list)
    for r in scored:
        per[r["category"]].append(r)

    # verification-as-reliability cross-tab: does it flag the weak answers, spare the strong ones?
    flagged = lambda r: r["verify_verdict"] in ("WARNING", "BLOCK")
    weak = [r for r in scored if r["score"] < 0.5]
    strong = [r for r in scored if r["score"] >= 0.5]
    caught = sum(1 for r in weak if flagged(r))
    false_alarm = sum(1 for r in strong if flagged(r))
    blocks = [r for r in rows if r["verify_verdict"] == "BLOCK"]

    L = []
    A = L.append
    tag = "Base + Security RAG" if data.get("rag") else "Base Moonlight"
    A(f"# Security Benchmark v3 (held-out) — {tag}\n")
    A(f"{data['model']} @ {data['model_revision'][:12]}. {len(rows)} held-out items, semantic "
      f"LLM-judge grading (rubric points present/absent + must-not penalty).\n")
    A(f"**Overall v3 score: {overall:.3f}**   over {len(scored)} scored"
      + (f"; {len(unscored)} UNSCORED (judge unparsed)" if unscored else "") + "\n")
    A("> " + CAVEAT + "\n")
    A("| Category | Score | items |")
    A("|---|--:|--:|")
    for cat in sorted(per):
        rs = per[cat]
        A(f"| {cat} | {sum(r['score'] for r in rs)/len(rs):.2f} | {len(rs)} |")
    A("\n## Verification layer (reliability, not capability)\n")
    A(f"- weak answers (score < 0.5): {len(weak)}; of those, verification flagged **{caught}**")
    A(f"- strong answers (score >= 0.5): {len(strong)}; verification flagged {false_alarm} "
      "(a flag on a strong answer is a citation/other note, not necessarily wrong)")
    A(f"- hard BLOCKs (arithmetic/code/phantom-action defects): {len(blocks)}"
      + (": " + ", ".join(r["id"] for r in blocks) if blocks else ""))
    A("- reminder: verification catches MECHANICAL defects, not subtle wrong reasoning — it "
      "complements the judge, it does not replace it.\n")
    if data.get("rag") and any("kept_after_gate" in r for r in rows):
        grounded = [r for r in rows if r.get("grounded")]
        A("## Relevance gate (retrieval quality)\n")
        A(f"- items where a passage survived the gate and grounded the answer: "
          f"{len(grounded)}/{len(rows)}")
        A("- the rest deferred to the model's own knowledge (gate found nothing directly relevant); "
          "on a strong base that is the safe default v3 established.\n")
    A("## Per-item\n")
    A("| id | category | score | verdict | grounded | detail |")
    A("|---|---|--:|---|:--:|---|")
    for r in rows:
        sc = "—" if r["score"] is None else f"{r['score']:.2f}"
        g = ("—" if not data.get("rag") else
             (f"{r.get('kept_after_gate', '?')}/{r.get('retrieved', '?')}"))
        A(f"| {r['id']} | {r['category']} | {sc} | {r['verify_verdict']} | {g} | {r['judge_detail']} |")
    c = data["cost"]
    A(f"\n## Cost\n- avg answer {c['mean_answer_tokens']:.0f} tok; "
      f"{c['mean_latency_s']} s/item (answer+judge); peak {c['peak_vram_gb']} GB")
    A("\n## Weakest answers")
    for r in sorted(scored, key=lambda r: r["score"])[:3]:
        A(f"- **{r['category']}** {r['id']} {r['score']:.2f}: {r['output'][:160].strip()}…")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", help="security index dir to retrieve context from")
    ap.add_argument("--model", help="model to run: alias (moonlight, qwen, qwen-14b) or a lock "
                    "file name; default = MODEL_LOCK env or the frozen Moonlight baseline")
    ap.add_argument("--report", help="rebuild report from an existing results.json (CPU)")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--judge-tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    if args.report:
        print(build_report(json.load(open(args.report, encoding="utf-8"))))
        return 0

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    from verification.verify import verify
    from rag.relevance import gate_relevance
    sys.path.insert(0, SECV3_DIR)
    from build_secv3 import grade_secv3, grade_deterministic, divergence

    if int(transformers.__version__.split(".")[0]) >= 5:
        sys.exit("transformers 5.x cannot quantise this model; install 4.57.6.")
    apply_all(verbose=False)
    from serving.model_spec import load_lock
    lock = load_lock(args.model)
    items = load_items()

    retriever = None
    if args.rag:
        from rag.store import Store
        from rag.query import build_prompt
        idx = args.rag if os.path.isabs(args.rag) else os.path.join(HERE, args.rag)
        retriever = (Store().load(idx), build_prompt)

    mode = "BASE + RAG" if args.rag else "BASE"
    print("=" * 74)
    print(f"SECURITY BENCHMARK v3 (HELD-OUT) — {mode} — {len(items)} items")
    print(CAVEAT)
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

    def generate(prompt: str, max_new: int) -> tuple[str, int, float]:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt").to(0)
        g0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = out[0][ids.shape[-1]:]
        return tok.decode(new, skip_special_tokens=True).strip(), int(new.shape[-1]), time.time() - g0

    torch.cuda.reset_peak_memory_stats()
    results = []
    for n, item in enumerate(items, 1):
        prompt, hits, retrieved_n = item["prompt"], None, 0
        if retriever:
            store, build_prompt = retriever
            hits = store.search(item["prompt"], k=args.k)
            retrieved_n = len(hits)
            # relevance gate: keep only passages that actually address THIS question. On held-out
            # topics this drops the tangential doc (the XXE 0.319 web-app case) so the model answers
            # from its strong own knowledge instead of being dragged wrong. One short judge call.
            hits = gate_relevance(item["prompt"], hits, lambda p: generate(p, 96)[0])
            prompt = build_prompt(item["prompt"], hits)

        answer, a_tok, a_lat = generate(prompt, args.max_new_tokens)

        # judge (same model, greedy) — closure hands grade_secv3 the raw judge text
        j_stat = {"tok": 0, "lat": 0.0}

        def judge_fn(jp: str) -> str:
            txt, jt, jl = generate(jp, args.judge_tokens)
            j_stat["tok"] += jt
            j_stat["lat"] += jl
            return txt

        score, detail = grade_secv3(item, answer, judge_fn)
        det = grade_deterministic(item, answer)          # independent keyword cross-check
        div = divergence(score, det)
        report = verify(answer, hits=hits, tools_ran=None)

        kept_n = len(hits) if hits else 0
        grounding = (f"{kept_n}/{retrieved_n} kept" if retriever else "no-rag")
        results.append({
            "id": item["id"], "domain": item["domain"], "category": item["category"],
            "score": score, "det_score": det, "divergence": div, "judge_detail": detail,
            "verify_verdict": report.verdict,
            "verify_findings": [str(f) for f in report.findings],
            "retrieved": retrieved_n, "kept_after_gate": kept_n,
            "grounded": kept_n > 0,
            "output": answer, "answer_tokens": a_tok,
            "latency_s": round(a_lat + j_stat["lat"], 2)})
        sc = "  —" if score is None else f"{score:.2f}"
        dsc = "  —" if det is None else f"{det:.2f}"
        flag = " ⚠DIVERGE" if (div is not None and div >= 0.34) else ""
        print(f"  [{n}/{len(items)}] {item['category']:14} judge={sc} det={dsc}{flag}  "
              f"{report.verdict:7}  [{grounding}]  {detail}")

    peak = torch.cuda.max_memory_allocated() / 1e9
    # tag results by MODEL so a swap does not clobber the baseline run (secv3_moonlight_base vs
    # secv3_qwen25-14b_base). Derived from the lock file name.
    mtag = (os.path.basename(lock.get("_lock_path", "MODEL_SPEC.lock.json"))
            .replace("MODEL_SPEC.", "").replace(".lock.json", "")) or "moonlight"
    data = {"name": f"secv3_{mtag}_" + ("rag" if args.rag else "base"), "model": lock["model"],
            "model_revision": lock["revision"], "rag": bool(args.rag), "items": len(items),
            "environment": {"transformers": transformers.__version__,
                            "device": torch.cuda.get_device_name(0),
                            "python": platform.python_version()},
            "cost": {"load_seconds": round(load_s, 1), "peak_vram_gb": round(peak, 2),
                     "mean_answer_tokens": sum(r["answer_tokens"] for r in results) / len(results),
                     "mean_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 2)},
            "results": results}
    out_dir = os.path.join(HERE, "evaluation", "results", data["name"])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(data, open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8"), indent=2)
    report_md = build_report(data)
    open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8").write(report_md)
    print("\n" + report_md)
    print(f"saved: evaluation/results/{data['name']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
