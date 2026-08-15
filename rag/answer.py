"""answer.py — the full Phase-1 assistant: retrieve -> ground -> generate with base Moonlight.

    python rag/answer.py "what attention does Moonlight use?"        # needs the GPU (Kaggle)

This is base Moonlight (untouched — no adapter, per the measured decision that SFT degraded it)
+ the capability_first policy layer + RAG. It:

  1. retrieves from the local index (CPU),
  2. builds a grounded, abstaining prompt (query.py),
  3. prepends the behaviour policy system prompt (serving/policy.py),
  4. generates with the base model.

Retrieval runs anywhere; generation needs the 4-bit model loaded, so run this where the GPU is.
The point of the whole arc: honesty and facts come from RETRIEVAL and a system prompt at
inference, not from baking examples into the weights.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):  # already reconfigured, or not a real stream
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--index", default=os.path.join(HERE, "rag", "index"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--policy", default=None, help="behaviour mode; default = config's active")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    # ---- retrieve + build the grounded prompt (CPU) --------------------------
    from rag.store import Store
    from rag.query import build_prompt, MIN_SCORE
    if not os.path.exists(os.path.join(args.index, "index_meta.json")):
        sys.exit(f"no index at {args.index} — run: python rag/ingest.py <docs_dir>")
    hits = Store().load(args.index).search(args.question, k=args.k)
    grounded = build_prompt(args.question, hits)
    best = hits[0]["score"] if hits else 0.0
    print(f"retrieved {len(hits)} chunks, best score {best:.2f} "
          f"-> {'grounded' if best >= MIN_SCORE else 'ABSTAIN'}")

    # ---- policy system prompt (serving/policy.py) ----------------------------
    from serving.policy import system_prompt
    sys_prompt = system_prompt(args.policy)

    # ---- generate with base Moonlight (GPU) ----------------------------------
    import torch
    if not torch.cuda.is_available():
        print("\nNO GPU — retrieval and prompt assembly done; run where the model is loaded to "
              "generate. Assembled prompt:\n")
        print(grounded)
        return 0

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    if int(transformers.__version__.split(".")[0]) >= 5:
        sys.exit("transformers 5.x cannot quantise this model; install 4.57.6.")
    apply_all(verbose=False)

    lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))
    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    print("loading base Moonlight (4-bit, no adapter)...")
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
    model.eval()
    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    messages = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + \
               [{"role": "user", "content": grounded}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt").to(0)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    answer = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()

    print("\n" + "=" * 74)
    print("ANSWER (base Moonlight, grounded in your documents):")
    print("=" * 74)
    print(answer)
    if best >= MIN_SCORE:
        print(f"\nsources: {', '.join(sorted({h['source'] for h in hits if h['score']>=MIN_SCORE}))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
