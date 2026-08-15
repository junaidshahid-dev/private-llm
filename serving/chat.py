"""chat.py — load Moonlight ONCE and talk to it interactively (from a notebook kernel).

    from serving.chat import Assistant
    bot = Assistant()                       # loads the model once (~8 min); then it stays resident
    print(bot.ask("what attention does Moonlight use?"))
    print(bot.ask("explain SSRF and how to prevent it", use_rag=False))

Use this from a Kaggle CODE cell, NOT a `!` shell cell. A `!python ...` cell spawns a fresh
process and reloads the whole 8.5GB model every time. Loaded in the kernel, the model stays in
memory and each question is just generation — seconds to a minute, no reload.

It is base Moonlight + the capability_first policy + (optional) RAG grounding — the same assistant
as rag/answer.py, but persistent so you can hold a conversation.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class Assistant:
    def __init__(self, index: str | None = None, policy: str | None = None, verbose: bool = True):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from training.patches import apply_all
        if int(transformers.__version__.split(".")[0]) >= 5:
            raise SystemExit("transformers 5.x cannot quantise this model; install 4.57.6.")
        apply_all(verbose=False)

        self.torch = torch
        lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))
        q = lock["quantization"]
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=q["double_quant"],
                                 bnb_4bit_compute_dtype=torch.float16)
        if verbose:
            print("loading Moonlight (4-bit) — one time, ~8 min if weights aren't cached...")
        self.model = AutoModelForCausalLM.from_pretrained(
            lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                                 trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # policy system prompt (capability_first by default)
        try:
            from serving.policy import system_prompt
            self.sys_prompt = system_prompt(policy)
        except Exception:                                    # noqa: BLE001
            self.sys_prompt = ""

        # optional RAG index
        self.store = None
        idx = index or os.path.join(HERE, "rag", "index")
        if os.path.exists(os.path.join(idx, "index_meta.json")):
            try:
                from rag.store import Store
                self.store = Store().load(idx)
                if verbose:
                    print(f"RAG index loaded ({idx})")
            except Exception as e:                           # noqa: BLE001
                if verbose:
                    print(f"(no RAG: {e})")
        if verbose:
            print("ready. ask with:  bot.ask(\"your question\")")

    def ask(self, question: str, use_rag: bool = True, max_new_tokens: int = 512, k: int = 5,
            show_sources: bool = True) -> str:
        prompt, sources = question, []
        if use_rag and self.store is not None:
            from rag.query import build_prompt, MIN_SCORE
            hits = self.store.search(question, k=k)
            if hits and hits[0]["score"] >= MIN_SCORE:
                prompt = build_prompt(question, hits)
                sources = sorted({h["source"] for h in hits if h["score"] >= MIN_SCORE})

        messages = ([{"role": "system", "content": self.sys_prompt}] if self.sys_prompt else []) \
            + [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt").to(0)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        answer = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        if show_sources and sources:
            answer += "\n\n[sources: " + ", ".join(sources) + "]"
        return answer
