"""test_webui.py — the local UI backend, incl. the E2E proving it drives the REAL agent.

    python tests/api/test_webui.py

Uses FastAPI's TestClient + a clearly-labelled stub model (mocks are allowed in tests). The E2E sends
a chat over the WebSocket, receives the model's tool PROPOSAL, APPROVES it from the "browser", and
confirms the REAL source_scan tool ran through the REAL controller + trust boundary + verifier — i.e.
the UI is connected to the actual architecture, not a fake.
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
import tempfile
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
os.environ["KILL_SWITCH_FILE"] = os.path.join(tempfile.mkdtemp(), ".KILL_SWITCH")  # isolate the switch

from fastapi.testclient import TestClient                                     # noqa: E402
from webui import model, server                                             # noqa: E402
from mcp_layer import killswitch                                            # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def drain_until(wsk, terminal, on_approval=None, cap=60):
    """Receive events until a terminal type; optionally auto-approve proposals. Returns all events."""
    got = []
    for _ in range(cap):
        m = wsk.receive_json()
        got.append(m)
        if m.get("type") == "approval_required" and on_approval:
            on_approval(m)
        if m.get("type") in terminal:
            break
    return got


def main() -> int:
    print("=" * 74)
    print("LOCAL UI BACKEND — API + E2E (UI -> real agent -> tool -> trust boundary -> verify)")
    print("=" * 74)
    killswitch.clear(operator_ack=True)
    client = TestClient(server.app)

    print("\n1. API starts and reports honestly")
    st = client.get("/api/status").json()
    check("status endpoint responds", "model" in st and "tools" in st)
    check("tool count is the real 34", st["tools"] == 34, str(st["tools"]))
    check("reports CUDA honestly (False on this CPU host)", st["device"]["cuda"] is False)
    check("tools endpoint lists tools", len(client.get("/api/tools").json()["tools"]) == 34)
    check("profiles endpoint lists real profiles",
          "recon" in client.get("/api/profiles").json()["profiles"])

    print("\n2. no model loaded -> chat errors honestly (never fakes a reply)")
    with client.websocket_connect("/ws") as wsk:
        wsk.send_json({"action": "chat", "prompt": "hello"})
        m = wsk.receive_json()
        check("errors with a reason when no model is loaded",
              m["type"] == "error" and m.get("reason"), str(m))

    print("\n3. session start is operator-gated + profile-enforced")
    r = client.post("/api/session/start", json={"objective": "assess", "targets": ["lab.local"],
                                                "profile": "recon"}).json()
    check("operator can start a session (the human at localhost)", r["ok"])
    bad = client.post("/api/session/start", json={"objective": "x", "targets": [],
                                                  "profile": "godmode"})
    check("an unknown profile is rejected", bad.status_code == 400)

    # load the clearly-labelled stub so the loop can run without a GPU
    model.use_stub()
    check("stub model is labelled a stub (not a real model)", model.status()["status"] == "stub")

    print("\n4. E2E: chat -> real tool PROPOSAL -> human APPROVE -> real execution -> verify")
    with client.websocket_connect("/ws") as wsk:
        wsk.send_json({"action": "chat", "prompt": "review the source of mcp_layer/security.py",
                       "history": []})
        approved = {"id": None}

        def do_approve(m):
            approved["id"] = m["id"]
            wsk.send_json({"action": "approve", "id": m["id"]})
        events = drain_until(wsk, {"completed", "error"}, on_approval=do_approve)
        kinds = [e["type"] for e in events]
        tkinds = [e.get("kind") for e in events if e["type"] == "telemetry"]
        check("a tool proposal was surfaced for approval", approved["id"] is not None, str(kinds))
        check("the REAL tool executed after approval (telemetry tool_result)",
              "tool_result" in tkinds, str(tkinds))
        check("verification ran on the real path", "verification" in tkinds, str(tkinds))
        check("the turn completed with a final answer", "completed" in kinds)

    print("\n5. DENY path: an unapproved tool does not execute")
    with client.websocket_connect("/ws") as wsk:
        wsk.send_json({"action": "chat", "prompt": "review the source of mcp_layer/tools.py"})

        def do_deny(m):
            wsk.send_json({"action": "deny", "id": m["id"]})
        ev = drain_until(wsk, {"completed", "error"}, on_approval=do_deny)
        auth = [e for e in ev if e.get("type") == "telemetry" and e.get("kind") == "authorization"]
        check("the denied proposal is recorded as declined",
              any(a.get("allowed") is False for a in auth), str(auth))

    print("\n6. kill switch halts a turn (backend-enforced, not cosmetic)")
    killswitch.engage("test")
    with client.websocket_connect("/ws") as wsk:
        wsk.send_json({"action": "chat", "prompt": "review the source of mcp_layer/security.py"})
        ev = drain_until(wsk, {"completed", "error", "blocked"})
        check("chat is blocked while the kill switch is engaged",
              any(e["type"] == "blocked" for e in ev), str([e["type"] for e in ev]))
    check("resume clears the kill switch", client.post("/api/session/resume").json()["ok"])

    print("\n7. malformed model output does not crash the backend")
    def _garbage(messages, max_new=768):
        return "{{{ not json and not a real answer 2+2=5"
    model._STATE.update(generate=_garbage, status="stub")
    with client.websocket_connect("/ws") as wsk:
        wsk.send_json({"action": "chat", "prompt": "hi"})
        ev = drain_until(wsk, {"completed", "error"})
        check("a garbage reply still completes with a verdict",
              any(e["type"] == "completed" for e in ev))

    print("\n" + "=" * 74)
    killswitch.clear(operator_ack=True)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL LOCAL-UI TESTS PASS — the UI drives the REAL agent (tool exec, trust boundary,")
    print("verification, kill switch, operator approval), never a fake.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
