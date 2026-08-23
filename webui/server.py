"""server.py — the local FastAPI adapter between the browser and the REAL agent.

Binds to 127.0.0.1 only. Every endpoint reuses the existing modules (session_policy, killswitch,
session/run_session, tools schema, verification, findings, report, memory) — it is a client of the
agent, never a reimplementation. The browser is a UI; the backend stays authoritative: approvals are
the human gate, execute_proposal still checks operator_ack + kill switch, tools stay gated, and the
trust boundary + verifier run on the real path.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DATA = os.path.join(REPO, "data", "webui")
os.makedirs(os.path.join(DATA, "conversations"), exist_ok=True)
os.makedirs(os.path.join(DATA, "reports"), exist_ok=True)

from webui import model                                                       # noqa: E402
from webui.runner import Turn                                                # noqa: E402

app = FastAPI(title="Local LLM Security Agent UI")
STATE = {"session": None}                          # the active AuthorizedSession (operator-started)


def _config():
    from mcp_layer import permissions as perm
    return perm.load_config()


def _tool_list():
    from mcp_layer import tools as toolmod, security as sec
    from web import tools as webmod
    return toolmod.schema() + sec.schema() + webmod.schema()


# ---- REST --------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/status")
def status():
    from mcp_layer import killswitch
    rag_ready = os.path.isdir(os.path.join(REPO, "rag", "index"))
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu"
    except Exception:                                # noqa: BLE001
        cuda, dev = False, "cpu"
    return {
        "model": model.status(),
        "device": {"cuda": cuda, "name": dev},
        "tools": len(_tool_list()),
        "rag": "ready" if rag_ready else "not_built",
        "memory": "ready",
        "verification": "ready",
        "trust_boundary": "ready",
        "kill_switch": "engaged" if killswitch.is_engaged() else "clear",
        "session": _session_view(),
    }


@app.get("/api/tools")
def tools():
    return {"tools": sorted(_tool_list(), key=lambda t: t["name"])}


@app.get("/api/profiles")
def profiles():
    from mcp_layer import session_policy
    return {"profiles": list(session_policy.PROFILES), "default": session_policy.DEFAULT_PROFILE}


def _session_view():
    s = STATE["session"]
    if not s:
        return None
    return {"id": s.id, "objective": s.objective, "targets": s.targets,
            "profile": s.capability_profile, "remaining_s": s.remaining_s()}


@app.get("/api/session")
def get_session():
    return {"session": _session_view()}


@app.post("/api/session/start")
async def session_start(body: dict):
    from mcp_layer import session_policy
    r = session_policy.start_session(
        body.get("objective", "assessment"), body.get("targets", []),
        body.get("profile", session_policy.DEFAULT_PROFILE),
        int(body.get("time_limit", 3600)),
        operator_ack=True,                            # the operator at localhost is the human gate
        config=_config())
    if not r["ok"]:
        return JSONResponse({"ok": False, "error": r["error"]}, status_code=400)
    STATE["session"] = r["session"]
    return {"ok": True, "session": _session_view()}


@app.post("/api/session/stop")
async def session_stop(body: dict | None = None):
    from mcp_layer import killswitch
    killswitch.engage((body or {}).get("reason", "operator stop from UI"), by="operator-ui")
    return {"ok": True, "kill_switch": "engaged"}


@app.post("/api/session/resume")
async def session_resume():
    from mcp_layer import killswitch
    return {"ok": killswitch.clear(operator_ack=True).get("ok"), "kill_switch": "clear"}


@app.post("/api/session/end")
async def session_end():
    STATE["session"] = None
    return {"ok": True}


@app.get("/api/findings")
def findings():
    return {"findings": STATE.get("last_findings", [])}


@app.get("/api/memory")
def memory():
    try:
        from memory.store import MemoryStore
        s = STATE["session"]
        project = (s.targets[0] if s and s.targets else "private-llm")
        store = MemoryStore(project=project)
        active = store._active(project)
        return {"project": project, "count": len(active),
                "items": [{"type": m["type"], "text": m["text"][:160]} for m in active[:10]]}
    except Exception as e:                            # noqa: BLE001
        return {"project": None, "count": 0, "error": str(e), "items": []}


@app.get("/api/reports")
def reports():
    d = os.path.join(DATA, "reports")
    return {"reports": sorted(os.listdir(d), reverse=True) if os.path.isdir(d) else []}


@app.get("/api/report/{name}")
def report(name: str):
    p = os.path.join(DATA, "reports", os.path.basename(name))
    if not os.path.isfile(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"name": name, "content": open(p, encoding="utf-8").read()}


@app.get("/api/conversations")
def conversations():
    d = os.path.join(DATA, "conversations")
    return {"conversations": sorted(os.listdir(d), reverse=True) if os.path.isdir(d) else []}


# ---- WebSocket: one chat turn over the real agent, with live events + approval ----------------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    current = {"turn": None}

    def emit(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    async def sender():
        while True:
            ev = await q.get()
            await websocket.send_json(ev)

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")
            if action == "chat":
                gen = model.get_generate()
                if gen is None:
                    st = model.status()
                    await websocket.send_json({"type": "error",
                        "reason": st.get("reason") or "no model loaded (start with --stub or a GPU host)"})
                    continue
                s = STATE.get("session")             # scope memory to the session's target
                proj = (s.targets[0] if s and getattr(s, "targets", None) else "private-llm")
                turn = Turn(msg.get("prompt", ""), gen, _config(), emit,
                            history=msg.get("history"), memory_project=proj)
                current["turn"] = turn

                def _run(t=turn):
                    rec = t.run()
                    _collect_findings(rec)
                loop.run_in_executor(None, _run)
            elif action in ("approve", "deny"):
                t = current["turn"]
                if t:
                    t.resolve(msg.get("id"), action == "approve")
            elif action == "stop":
                from mcp_layer import killswitch
                killswitch.engage("operator stop from UI", by="operator-ui")
                if current["turn"]:
                    current["turn"].cancel_pending()
                await websocket.send_json({"type": "session_stopped"})
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        if current["turn"]:
            current["turn"].cancel_pending()


def _collect_findings(record):
    """Turn any tool findings (source_scan etc.) into the findings view + save a report."""
    if not record or not record.get("results"):
        return
    from serving.autonomous import _hyp_from_finding
    from research.report import assessment_report
    hyps = []
    for r in record["results"]:
        for f in ((r.get("result") or {}).get("findings") or []):
            hyps.append(_hyp_from_finding(f))
    if not hyps:
        return
    STATE["last_findings"] = [{"title": h.title, "severity": h.severity, "status": h.status,
                               "component": h.affected_component, "vuln_class": h.vuln_class,
                               "next_test": h.next_test} for h in hyps]
    rep = assessment_report(objective="chat assessment",
                            scope=(STATE["session"].targets if STATE["session"] else []),
                            findings=hyps)
    name = f"report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    with open(os.path.join(DATA, "reports", name), "w", encoding="utf-8") as f:
        f.write(rep)
