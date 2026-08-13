"""prepare_dataset.py — raw -> clean -> dedup -> filter -> tokenize -> split.

Runs on a laptop. No GPU, no model weights; only the tokenizer is fetched, and even that is
optional (see TOKENIZER below).

    python training/prepare_dataset.py
    python training/prepare_dataset.py --no-tokenizer     # skip token stats, use char estimate

WHY THE ORDER MATTERS

Deduplication happens BEFORE splitting. Doing it after is the classic contamination bug: a
duplicated example lands in both train and test, the model memorises it, and the eval reports an
improvement that is really recall. This script also checks explicitly for overlap after
splitting and refuses to write if it finds any.

The held-out EVALUATION SET IS NOT PRODUCED HERE. evaluation/benchmark.jsonl is written once, by
hand, before any training, and never regenerated from training data. A benchmark that moves with
the training set cannot measure whether training helped.

INPUT FORMAT — one JSON object per line in data/raw/*.jsonl, either:

    {"instruction": "...", "input": "(optional)", "output": "..."}
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Both normalise to the same internal shape, so you can mix sources.

ON DATA QUALITY. A 16B model fine-tuned on 500 careful examples beats the same model fine-tuned
on 50,000 scraped ones. LoRA does not teach facts; it teaches shape - format, register, how a
refusal is phrased, how a tool call is structured. Every bad example teaches a bad shape, and
the model has no way to tell your mistakes from your intentions.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(HERE, "data", *p)          # noqa: E731
SEED = 20260813
SPLIT = (0.90, 0.05, 0.05)                              # train / validation / test

MIN_OUTPUT_CHARS = 20
MAX_OUTPUT_CHARS = 24000
MIN_PROMPT_CHARS = 8
MAX_TOKENS = 1024                                       # must match configs max_seq_len


def norm(s: str) -> str:
    """Aggressive normalisation, used ONLY for duplicate detection, never for output."""
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def load_raw() -> list[dict]:
    rows, files = [], sorted(glob.glob(D("raw", "*.jsonl")))
    if not files:
        raise SystemExit(f"no files in {D('raw')} — add at least one .jsonl")
    for path in files:
        src = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append({"_src": src, "_line": n, **json.loads(line)})
                except json.JSONDecodeError as e:
                    print(f"  malformed JSON {src}:{n} — dropped ({e.msg})", file=sys.stderr)
    return rows


def to_messages(r: dict) -> list[dict] | None:
    """Normalise both accepted shapes into a message list."""
    if isinstance(r.get("messages"), list) and r["messages"]:
        msgs = [m for m in r["messages"]
                if isinstance(m, dict) and m.get("role") and isinstance(m.get("content"), str)]
        return msgs or None
    if r.get("instruction") and r.get("output"):
        user = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": r["output"]}]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()

    for sub in ("raw", "cleaned", "processed", "train", "validation", "test"):
        os.makedirs(D(sub), exist_ok=True)

    raw = load_raw()
    print(f"raw            {len(raw):>6} examples from {len(set(r['_src'] for r in raw))} files")
    stats = Counter()

    # ---- clean ---------------------------------------------------------------
    cleaned = []
    for r in raw:
        msgs = to_messages(r)
        if not msgs:
            stats["dropped: unrecognised shape"] += 1
            continue
        if msgs[-1]["role"] != "assistant":
            stats["dropped: no assistant turn last"] += 1
            continue
        prompt = " ".join(m["content"] for m in msgs[:-1])
        answer = msgs[-1]["content"]
        if len(prompt.strip()) < MIN_PROMPT_CHARS:
            stats["dropped: prompt too short"] += 1
            continue
        if not (MIN_OUTPUT_CHARS <= len(answer.strip()) <= MAX_OUTPUT_CHARS):
            stats["dropped: answer length"] += 1
            continue
        if norm(prompt) == norm(answer):
            stats["dropped: answer echoes prompt"] += 1
            continue
        cleaned.append({"messages": msgs, "_src": r["_src"], "_line": r["_line"]})
    print(f"cleaned        {len(cleaned):>6}")

    # ---- deduplicate (BEFORE splitting) --------------------------------------
    seen, deduped = {}, []
    for r in cleaned:
        key = hashlib.sha256(
            norm(" ".join(m["content"] for m in r["messages"])).encode()).hexdigest()
        if key in seen:
            stats["dropped: duplicate"] += 1
            continue
        seen[key] = True
        r["_hash"] = key
        deduped.append(r)
    print(f"deduplicated   {len(deduped):>6}")

    # ---- tokenize / length filter -------------------------------------------
    tok = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))
            tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                                trust_remote_code=True)
            print(f"tokenizer      {lock['model']} @ {lock['revision'][:8]}")
        except Exception as e:                                   # noqa: BLE001
            print(f"tokenizer      unavailable ({type(e).__name__}: {str(e)[:70]})")
            print("               falling back to a chars/3.6 estimate — RE-RUN before training,")
            print("               because the real token count decides what fits in 1024.")

    kept = []
    for r in deduped:
        text = "\n".join(m["content"] for m in r["messages"])
        n = len(tok(text)["input_ids"]) if tok else int(len(text) / 3.6)
        if n > MAX_TOKENS:
            stats[f"dropped: over {MAX_TOKENS} tokens"] += 1
            continue
        r["_tokens"] = n
        kept.append(r)
    lens = [r["_tokens"] for r in kept]
    print(f"length-filtered{len(kept):>6}"
          + (f"   tokens: mean {sum(lens)/len(lens):.0f}, max {max(lens)}" if lens else ""))

    if not kept:
        raise SystemExit("nothing survived filtering — check data/raw/")

    # ---- split ---------------------------------------------------------------
    rng = random.Random(SEED)
    rng.shuffle(kept)
    n = len(kept)
    a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    parts = {"train": kept[:a], "validation": kept[a:b], "test": kept[b:]}

    # ---- contamination check -------------------------------------------------
    tr = {r["_hash"] for r in parts["train"]}
    for name in ("validation", "test"):
        overlap = tr & {r["_hash"] for r in parts[name]}
        if overlap:
            raise SystemExit(f"CONTAMINATION: {len(overlap)} examples in both train and {name}")
    bench = os.path.join(HERE, "evaluation", "benchmark.jsonl")
    if os.path.exists(bench):
        bh = set()
        with open(bench, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    bh.add(hashlib.sha256(norm(o.get("prompt", "")).encode()).hexdigest())
        hit = [r for r in parts["train"]
               if hashlib.sha256(norm(r["messages"][0]["content"]).encode()).hexdigest() in bh]
        if hit:
            raise SystemExit(f"CONTAMINATION: {len(hit)} benchmark prompts appear in train")
        print(f"benchmark      {len(bh)} prompts, none present in train")

    # ---- write ---------------------------------------------------------------
    for name, rows in parts.items():
        with open(D(name, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

    version = hashlib.sha256(
        "".join(sorted(r["_hash"] for r in kept)).encode()).hexdigest()[:16]
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": version,
        "seed": SEED,
        "counts": {k: len(v) for k, v in parts.items()},
        "raw_examples": len(raw),
        "dropped": dict(stats),
        "tokenizer_used": bool(tok),
        "max_tokens": MAX_TOKENS,
        "token_stats": {"mean": sum(lens) / len(lens), "max": max(lens), "min": min(lens)},
        "sources": dict(Counter(r["_src"] for r in kept)),
    }
    with open(D("processed", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\ndropped:")
    for k, v in stats.most_common():
        print(f"  {v:>5}  {k}")
    print(f"\nsplit          train {len(parts['train'])} / "
          f"val {len(parts['validation'])} / test {len(parts['test'])}")
    print(f"dataset version {version}   (record this with every training run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
