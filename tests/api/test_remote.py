"""test_remote.py — the REMOTE-GPU backend: run the UI/agent locally, model inference over HTTP.

    python tests/api/test_remote.py

Two halves, both on CPU with a clearly-labelled stub (mocks are allowed in tests):
  1. serving/gpu_server.py's contract — POST /generate returns {text}; the X-Auth secret is enforced.
  2. webui.model.use_remote() — points the generate() seam at a REAL socket (a tiny mock model server),
     and a real run_session drives a tool through it: proving the local agent works with a remote model.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ---- a tiny mock MODEL server on a real socket (stands in for gpu_server on Kaggle) -------------
_CALLS = {"n": 0}


class _MockModel(BaseHTTPRequestHandler):
    def log_message(self, *a):                            # silence
        pass

    def _send(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "model": "mock-remote", "device": "mock GPU", "status": "ready"})
        else:
            self._send(404, {"error": "nope"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _CALLS["n"] += 1
        # first call: propose a tool (drives the agent loop); later calls: a plain final answer.
        if _CALLS["n"] == 1:
            text = ('I will statically review the source. '
                    '{"tool": "source_scan", "arguments": {"path": "mcp_layer/security.py"}, '
                    '"why": "look for dangerous sinks"}')
        else:
            text = "Reviewed the source; findings summarised."
        self._send(200, {"text": text})


def main() -> int:
    print("=" * 74)
    print("REMOTE-GPU BACKEND — UI/agent local, model over HTTP")
    print("=" * 74)

    from webui import model as webmodel
    from serving import gpu_server
    from fastapi.testclient import TestClient

    print("\n1. gpu_server contract — /generate returns {text}; /health reports; auth enforced")
    webmodel.use_stub()                                   # so the server has a generate() to serve
    client = TestClient(gpu_server.app)
    h = client.get("/health").json()
    check("health reports the served model", h["ok"] and h["status"] == "stub", str(h))
    r = client.post("/generate", json={"messages": [{"role": "user", "content": "hello there"}]})
    check("generate returns text", r.status_code == 200 and "hello there" in r.json()["text"], str(r.json()))
    # secret enforcement
    gpu_server._CFG["secret"] = "s3cret"
    no_auth = client.post("/generate", json={"messages": [{"role": "user", "content": "x"}]})
    check("missing X-Auth is 401 when a secret is set", no_auth.status_code == 401)
    with_auth = client.post("/generate", headers={"X-Auth": "s3cret"},
                            json={"messages": [{"role": "user", "content": "x"}]})
    check("correct X-Auth is accepted", with_auth.status_code == 200)
    gpu_server._CFG["secret"] = None

    print("\n2. use_remote() — the UI's generate() seam calls a REAL remote socket")
    srv = HTTPServer(("127.0.0.1", 0), _MockModel)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"

    st = webmodel.use_remote(url)
    check("use_remote reaches the server and reports status 'remote'", st["status"] == "remote", str(st))
    check("remote model name comes from /health", st["model"] == "mock-remote", str(st))
    bad = webmodel.use_remote("http://127.0.0.1:1")       # nothing listening
    check("an unreachable remote fails honestly (status error, no fake model)",
          bad["status"] == "error" and webmodel.get_generate() is None, str(bad))

    print("\n3. a REAL run_session drives a tool THROUGH the remote model")
    _CALLS["n"] = 0
    webmodel.use_remote(url)                               # re-point at the working mock
    gen = webmodel.get_generate()
    from mcp_layer.session import run_session
    calls = []

    def executor(proposal, config, operator_ack=False):
        calls.append(proposal.get("tool"))
        return {"ok": True, "tool": proposal.get("tool"), "result": {"ran": True}}

    rec = run_session("review the source of mcp_layer/security.py", gen,
                      approver=lambda p: True, executor=executor, config={})
    check("the remote model proposed a tool that the local agent executed",
          "source_scan" in calls and "source_scan" in rec["executed_tools"], str(calls))
    check("the session completed with a final answer from the remote model",
          bool(rec.get("final")), str(rec.get("final"))[:80])

    print("\n" + "=" * 74)
    srv.shutdown()
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL REMOTE-BACKEND TESTS PASS — the local UI/agent runs the loop with the model over HTTP,")
    print("so the GPU can live on Kaggle while everything else runs on your machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
