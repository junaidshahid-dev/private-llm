"""judge_pass.py — re-judge SAVED answers with an INDEPENDENT judge model (author != judge).

    python evaluation/judge_pass.py secv3_qwen25-coder-14b_base --judge-model moonlight
    python evaluation/judge_pass.py secv3_lock.json_base        --judge-model qwen

The model-swap experiment proved the self-judge is home-field biased (Moonlight rated its own
answers 1.000 while the independent deterministic grader said 0.675). The fix your architecture
asked for: don't let a model judge its own family. This loads the ANSWERS from a finished run (they
are already in results.json["output"]) and re-scores them with a DIFFERENT model — so Moonlight's
answers get judged by Qwen and vice-versa. Only ONE model is in memory at a time (answers are
pre-generated), so it fits a single T4.

Cross-judge both directions and the home-field bias cancels:
    judge_pass secv3_lock.json_base        --judge-model qwen       -> Moonlight answers, Qwen judge
    judge_pass secv3_qwen25-coder-14b_base --judge-model moonlight  -> Qwen answers, Moonlight judge
then compare the CROSS-judged overalls (plus the deterministic grader, which never changes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
SECV3_DIR = os.path.join(HERE, "evaluation", "development", "security_v3")
RESULTS = os.path.join(HERE, "evaluation", "results")


def resolve_results(spec: str) -> str:
    if os.path.isfile(spec):
        return spec
    for c in (os.path.join(spec, "results.json"), os.path.join(RESULTS, spec, "results.json")):
        if os.path.isfile(c):
            return c
    sys.exit(f"no results.json for {spec!r}")


def rejudge_rows(rows, items_by_id, judge_fn) -> list[dict]:
    """Re-score each saved answer with judge_fn. Deterministic grade is independent of the judge and
    recomputed for the divergence check. Testable with a scripted judge_fn (no model)."""
    sys.path.insert(0, SECV3_DIR)
    from build_secv3 import grade_secv3, grade_deterministic, divergence
    out = []
    for r in rows:
        item = items_by_id.get(r["id"])
        if not item:
            continue
        ans = r.get("output", "")
        score, detail = grade_secv3(item, ans, judge_fn)
        det = grade_deterministic(item, ans)
        out.append({**r, "score": score, "det_score": det, "divergence": divergence(score, det),
                    "judge_detail": detail})
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="a finished run: results.json path, dir, or secv3_<tag>_base")
    ap.add_argument("--judge-model", required=True, help="model to JUDGE with (alias/lock)")
    ap.add_argument("--judge-tokens", type=int, default=256)
    args = ap.parse_args()

    data = json.load(open(resolve_results(args.results), encoding="utf-8"))
    rows = data["results"]
    sys.path.insert(0, SECV3_DIR)
    from build_secv3 import items_as_dicts
    items = {d["id"]: d for d in items_as_dicts()}

    from serving.model_spec import load_lock
    lock = load_lock(args.judge_model)
    answer_model = data.get("model", "?")
    print("=" * 74)
    print(f"JUDGE PASS — answers by {answer_model.split('/')[-1]}, judged by {lock['model'].split('/')[-1]}")
    print("=" * 74)

    import torch
    if not torch.cuda.is_available():
        print("NO GPU — needs the judge model. Run on Kaggle.")
        return 2
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    apply_all(verbose=False)
    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    print(f"loading judge model {lock['model']} (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
    model.eval()
    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def judge_fn(prompt: str) -> str:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt").to(0)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=args.judge_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        return tok.decode(o[0][ids.shape[-1]:], skip_special_tokens=True).strip()

    t0 = time.time()
    rejudged = rejudge_rows(rows, items, judge_fn)
    for r in rejudged:
        sc = "  —" if r["score"] is None else f"{r['score']:.2f}"
        dv = " ⚠" if (r["divergence"] is not None and r["divergence"] >= 0.34) else ""
        print(f"  {r['id']:22} judge={sc} det={r['det_score']}{dv}  {r['judge_detail']}")

    jm = lock["model"].split("/")[-1]
    am = answer_model.split("/")[-1]
    print("\n" + "-" * 74)
    print(f"CROSS-JUDGED overall (judge={jm}): {_mean([r['score'] for r in rejudged]):.3f}")
    print(f"deterministic overall (unbiased): {_mean([r['det_score'] for r in rejudged]):.3f}")
    out = {**data, "name": f"secv3_{am}_judgedby_{jm}", "judge_model": lock["model"],
           "results": rejudged}
    d = os.path.join(RESULTS, out["name"])
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "results.json"), "w", encoding="utf-8"), indent=2)
    print(f"took {time.time()-t0:.0f}s. saved -> evaluation/results/{out['name']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
