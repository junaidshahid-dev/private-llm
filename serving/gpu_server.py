"""gpu_server.py — serve the model's generate() over HTTP so the LOCAL UI can use a REMOTE GPU.

The whole point: run the UI + agent + tools on your own machine, but let the heavy model inference run
on a Kaggle GPU. This is the model half of that — a tiny FastAPI server that loads the model exactly
the way the UI would (the same lock seam + optional fine-tuned adapter) and exposes one endpoint:

    POST /generate  {"messages": [{role,content}...], "max_new": 768}  ->  {"text": "..."}
    GET  /health    -> {ok, model, device, status}

Run it on the GPU host (Kaggle), expose it with a tunnel (cloudflared / ngrok), then point the local
UI at the tunnel URL:

    # on Kaggle (GPU):
    python serving/gpu_server.py --adapter models/experiment-001/final --secret mysecret
    #   ... then a tunnel gives you e.g. https://abc-123.trycloudflare.com

    # on your laptop:
    python start_local.py --remote https://abc-123.trycloudflare.com --remote-secret mysecret

It reuses webui.model.load(), so the base + adapter + 4-bit config are identical to a local GPU run.
The optional shared secret (X-Auth header) keeps the public tunnel from being an open model endpoint.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# reduce CUDA fragmentation before torch is imported (torch loads lazily inside webui.model.load)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI, Header, HTTPException                            # noqa: E402

app = FastAPI(title="LLM GPU inference server")
_CFG = {"secret": None}


@app.get("/health")
def health():
    from webui import model
    st = model.status()
    return {"ok": model.is_ready(), "model": st["model"], "device": st["device"],
            "status": st["status"]}


@app.post("/generate")
async def generate(body: dict, x_auth: str | None = Header(default=None)):
    from webui import model
    if _CFG["secret"] and x_auth != _CFG["secret"]:
        raise HTTPException(status_code=401, detail="bad or missing X-Auth")
    gen = model.get_generate()
    if gen is None:
        raise HTTPException(status_code=503,
                            detail=model.status().get("reason") or "model not loaded on the server")
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list of {role, content}")
    try:
        text = gen(messages, int(body.get("max_new", 768)))
    except Exception as e:                               # surface the real cause, not a blank 500
        import traceback
        traceback.print_exc()                            # full traceback to the server log
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return {"text": text}


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the model over HTTP for the local UI (remote GPU).")
    ap.add_argument("--host", default="0.0.0.0")       # 0.0.0.0 so a tunnel can reach it
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--model", default=None, help="model alias/lock (default: the working base, Qwen)")
    ap.add_argument("--adapter", default=None,
                    help="trained LoRA adapter dir to serve (e.g. models/experiment-001/final)")
    ap.add_argument("--secret", default=None,
                    help="shared secret required in the X-Auth header (protect the public tunnel)")
    ap.add_argument("--stub", action="store_true",
                    help="serve the clearly-labelled echo stub instead of a real model (no GPU) — "
                         "for wiring up the tunnel without loading weights")
    args = ap.parse_args()

    from webui import model
    st = model.use_stub() if args.stub else model.load(args.model, adapter=args.adapter)
    _CFG["secret"] = args.secret

    print("=" * 60)
    print("LLM GPU INFERENCE SERVER (for a remote local UI)")
    print("=" * 60)
    print(f"  model:   {st.get('model') or '—'}  [{st['status']}]")
    print(f"  device:  {st.get('device') or 'cpu'}")
    print(f"  auth:    {'X-Auth secret required' if args.secret else 'OPEN (no secret) — add --secret'}")
    if st["status"] not in ("ready", "stub"):
        print(f"\n  NOTE: {st.get('reason')}")
        print("  The server will still start but /generate returns 503 until a model is loaded.")
        print("  On a CPU host use --stub; on a GPU host omit it.")
    print(f"\n  serving on {args.host}:{args.port}")
    print("  expose it with a tunnel, e.g.:")
    print(f"    cloudflared tunnel --url http://localhost:{args.port}")
    print("  then on your laptop:")
    print("    python start_local.py --remote <tunnel-url>"
          + (" --remote-secret <secret>" if args.secret else ""))
    print("=" * 60)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
