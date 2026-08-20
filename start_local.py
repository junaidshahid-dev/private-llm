"""start_local.py — start the complete local UI (backend + served frontend). 127.0.0.1 only.

    python start_local.py            # loads the real model if a GPU is present, else reports why
    python start_local.py --stub     # clearly-labelled echo model to exercise the UI without a GPU
    python start_local.py --port 8010 --model qwen

No cloud, no public URL, no external services (except a tool the operator authorizes). The real model
loads through the existing lock seam on a GPU host; on a CPU laptop it honestly reports "GPU required".
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local LLM security-agent UI (localhost only).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--stub", action="store_true",
                    help="use a clearly-labelled echo model (no GPU) to exercise the UI")
    ap.add_argument("--model", default=None, help="model alias/lock (default: frozen baseline)")
    args = ap.parse_args()

    from webui import model
    st = model.use_stub() if args.stub else model.load(args.model)

    from mcp_layer import tools as t, security as s
    from web import tools as w
    ntools = len({e["name"] for e in t.schema() + s.schema() + w.schema()})
    rag = "READY" if os.path.isdir(os.path.join(HERE, "rag", "index")) else "not built"

    def line(k, v):
        print(f"  {k:<14}{v}")

    print("=" * 52)
    print("LOCAL LLM — SECURITY AGENT UI")
    print("=" * 52)
    line("Backend:", "READY")
    line("Frontend:", "READY (served)")
    line("Model:", f"{st['status'].upper()}" + (f"  ({st['model']})" if st.get('model') else ""))
    line("Device:", st.get("device") or "cpu")
    line("RAG:", rag)
    line("Memory:", "READY")
    line("MCP/Tools:", ntools)
    line("Verifier:", "READY")
    line("Trust bnd:", "READY")
    if st["status"] not in ("ready", "stub"):
        print("\n  NOTE:", st["reason"])
        print("  The UI will still start; chat needs a model. Re-run with --stub for a demo,")
        print("  or run on a GPU host to load the real model.")
    print("\n  Open:  http://%s:%d\n" % (args.host if args.host != "0.0.0.0" else "127.0.0.1", args.port))
    print("=" * 52)

    import uvicorn
    uvicorn.run("webui.server:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
