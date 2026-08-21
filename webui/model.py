"""model.py — the generate() seam for the local UI. Real model on a GPU host; honest on CPU.

The whole agent (run_session/run_assessment) already takes an injectable `generate(messages)->str`,
so the UI does not change the model code — it loads the real model through the EXISTING seam
(serving.model_spec.load_lock + the same 4-bit config the benchmarks use) when a CUDA GPU is present,
and otherwise reports the truth: "GPU required". A clearly-labelled echo stub is available only when
the operator explicitly asks (--stub), for exercising the UI without a GPU; it never masquerades as a
real model (the status says "stub").
"""
from __future__ import annotations

import os
import threading

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE = {"generate": None, "status": "not_loaded", "model": None, "device": None, "reason": ""}
_LOCK = threading.Lock()


def status() -> dict:
    return {k: _STATE[k] for k in ("status", "model", "device", "reason")}


def is_ready() -> bool:
    return _STATE["generate"] is not None


def get_generate():
    return _STATE["generate"]


def use_stub() -> dict:
    """Opt-in, clearly-labelled echo generator so the UI can be exercised without a GPU. NOT a model:
    status is 'stub' and every reply is prefixed so it can never be mistaken for real output."""
    def gen(messages, max_new=768):
        last = messages[-1]["content"] if messages else ""
        # if the operator's message hints at a tool, propose a read-only one so the loop is exercised
        low = last.lower()
        if "scan" in low and "source" in low or "review the source" in low or "analyze the code" in low:
            return ('I will statically review the source. '
                    '{"tool": "source_scan", "arguments": {"path": "mcp_layer/security.py"}, '
                    '"why": "look for dangerous sinks"}')
        return f"[STUB MODEL — not a real model] You said: {last[:600]}"
    with _LOCK:
        _STATE.update(generate=gen, status="stub", model="stub/echo", device="cpu",
                      reason="clearly-labelled demo stub; not a real model — start without --stub on a "
                             "GPU host to load the real model")
    return status()


def use_remote(url: str, secret: str | None = None) -> dict:
    """Use a REMOTE model server (serving/gpu_server.py behind a tunnel) as the generate() backend.
    The UI, agent, tools, verifier and kill switch all still run locally; only the model inference is
    remote. This is how you run everything on your own machine but put the GPU on Kaggle."""
    import json
    import urllib.error
    import urllib.request
    base = (url or "").rstrip("/")
    hdr = {"Content-Type": "application/json"}
    if secret:
        hdr["X-Auth"] = secret
    # Probe /health so a bad URL/secret fails fast and honestly at startup, not on the first message.
    try:
        req = urllib.request.Request(base + "/health", headers=hdr)
        with urllib.request.urlopen(req, timeout=30) as r:
            h = json.loads(r.read())
    except Exception as e:                                # noqa: BLE001
        with _LOCK:
            _STATE.update(status="error", generate=None,
                          reason=f"cannot reach remote model at {base}: {type(e).__name__}: {e}")
        return status()

    def gen(messages, max_new=768):
        body = json.dumps({"messages": messages, "max_new": max_new}).encode()
        req = urllib.request.Request(base + "/generate", data=body, headers=hdr, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read()).get("text", "")
        except urllib.error.HTTPError as e:              # surface the server's real error, not a blank 500
            try:
                detail = json.loads(e.read() or b"{}").get("detail", "")
            except Exception:                            # noqa: BLE001
                detail = ""
            raise RuntimeError(f"remote model error {e.code}: {detail or e.reason}")

    with _LOCK:
        _STATE.update(generate=gen, status="remote", model=(h.get("model") or "remote model"),
                      device=f"remote GPU ({base})", reason="")
    return status()


def load(model_alias: str | None = None, adapter: str | None = None) -> dict:
    """Load the real model via the existing lock seam if a CUDA GPU is present; else report why not.
    If `adapter` names a trained LoRA dir (e.g. models/experiment-001/final), it is applied on top of
    the base — this is how the fine-tuned security model is served in the UI."""
    with _LOCK:
        if _STATE["generate"] is not None and _STATE["status"] in ("ready", "stub"):
            return status()
        try:
            import torch
        except ImportError:
            _STATE.update(status="unavailable", reason="torch is not installed")
            return status()
        if not torch.cuda.is_available():
            _STATE.update(status="gpu_required", device="cpu", model=None,
                          reason="no CUDA GPU on this host — the model needs a GPU. Start with --stub "
                                 "to exercise the UI, or run on a GPU host (e.g. Kaggle).")
            return status()
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from training.patches import apply_all
            from serving.model_spec import load_lock
            if int(transformers.__version__.split(".")[0]) >= 5:
                _STATE.update(status="error", reason="transformers 5.x cannot quantise this model; "
                              "install 4.57.6")
                return status()
            apply_all(verbose=False)
            lock = load_lock(model_alias)
            q = lock["quantization"]
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=q["double_quant"],
                                     bnb_4bit_compute_dtype=torch.float16)
            # Spread across all GPUs when there's more than one (e.g. Kaggle T4 x2). A 14B model's
            # weights (~10GB 4-bit) plus the attention buffer for a long system prompt do NOT fit on a
            # single 15GB T4 — "balanced_low_0" keeps GPU 0 light so the forward-pass buffers have room.
            ndev = torch.cuda.device_count()
            device_map = "balanced_low_0" if ndev > 1 else {"": 0}
            model = AutoModelForCausalLM.from_pretrained(
                lock["model"], revision=lock["revision"], quantization_config=bnb, device_map=device_map)
            label = lock["model"] + (f"  ({ndev}× {torch.cuda.get_device_name(0)})" if ndev > 1 else "")
            if adapter:
                from peft import PeftModel
                ad = adapter if os.path.isabs(adapter) else os.path.join(_REPO, adapter)
                if not os.path.isdir(ad):
                    _STATE.update(status="error", reason=f"adapter dir not found: {ad}")
                    return status()
                model = PeftModel.from_pretrained(model, ad)
                label = f"{lock['model']} + {os.path.basename(os.path.dirname(ad.rstrip('/\\\\')))}"
            model.eval()
            tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                                trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            def gen(messages, max_new=768):
                ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                              return_tensors="pt").to(0)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
            _STATE.update(generate=gen, status="ready", model=label,
                          device=torch.cuda.get_device_name(0), reason="")
        except Exception as e:                        # noqa: BLE001
            _STATE.update(status="error", reason=f"{type(e).__name__}: {e}")
        return status()
