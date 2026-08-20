"""runner.py — drive the REAL agent for one chat turn, streaming live events + human approval.

This is the adapter, not a reimplementation: it calls the existing mcp_layer.session.run_session with
the real controller/tools/verification/trust-boundary, and uses the existing Telemetry ledger as the
event bus (its records — plan/proposal/authorization/tool_result/interpretation/verification/report —
ARE the live activity feed). Tool proposals block on a human decision made in the browser (the operator
at localhost is the human gate); the existing execute_proposal still enforces operator_ack + the kill
switch independently, so the frontend can never execute a tool on its own. Kill switch halts a turn
immediately via the existing killswitch. No FastAPI here, so it is unit-testable.
"""
from __future__ import annotations

import threading
import uuid

APPROVAL_TIMEOUT = 900          # seconds a proposal waits for the human before it is auto-denied


class Turn:
    """One chat turn = one run_session over the real agent, with events + human approval."""

    def __init__(self, question, generate, config, emit, history=None):
        self.question = question
        self.generate = generate
        self.config = config
        self.emit = emit                              # emit(event: dict) -> None
        self.history = history or []
        self.pending: dict = {}                       # approval_id -> {event, decision, proposal}
        self.record = None

    # ---- human approval (blocks the loop until the browser decides) -------------------------
    def _approver(self, proposal) -> bool:
        aid = uuid.uuid4().hex[:8]
        ev = threading.Event()
        self.pending[aid] = {"event": ev, "decision": None, "proposal": proposal}
        args = proposal.get("arguments") or {}
        self.emit({"type": "approval_required", "id": aid, "tool": proposal.get("tool"),
                   "target": args.get("target") or args.get("url") or args.get("path") or "",
                   "arguments": args, "why": proposal.get("why", "")})
        ev.wait(timeout=APPROVAL_TIMEOUT)
        decided = self.pending[aid]["decision"] is True
        if not ev.is_set():
            self.emit({"type": "approval_timeout", "id": aid})
        return decided                                # operator_ack for execute_proposal is this bool

    def resolve(self, approval_id, decision) -> bool:
        p = self.pending.get(approval_id)
        if not p or p["event"].is_set():
            return False
        p["decision"] = bool(decision)
        p["event"].set()
        return True

    def cancel_pending(self):
        for p in self.pending.values():
            if not p["event"].is_set():
                p["decision"] = False
                p["event"].set()

    # ---- the turn ---------------------------------------------------------------------------
    def run(self):
        from mcp_layer import killswitch
        from mcp_layer.session import run_session
        from mcp_layer.telemetry import Telemetry
        from serving.policy import system_prompt

        if killswitch.is_engaged():
            self.emit({"type": "blocked", "reason": "kill switch engaged — clear it to run"})
            self.emit({"type": "completed", "final": "", "verification": None})
            return {"final": "", "halted": "kill switch"}

        tel = Telemetry("ui", sink=lambda rec: self.emit({"type": "telemetry", **rec}))
        self.emit({"type": "thinking"})
        q = self._with_history(self.question)
        try:
            self.record = run_session(q, self.generate, approver=self._approver, config=self.config,
                                      policy_prompt=system_prompt(), telemetry=tel)
        except Exception as e:                        # noqa: BLE001 — surface, never crash the server
            self.emit({"type": "error", "reason": f"{type(e).__name__}: {e}"})
            return {"final": "", "error": str(e)}
        self.emit({"type": "completed", "final": self.record.get("final", ""),
                   "verification": self.record.get("verification"),
                   "executed_tools": self.record.get("executed_tools", []),
                   "escalated": self.record.get("escalated"),
                   "proposals": self.record.get("proposals", [])})
        return self.record

    def _with_history(self, question: str) -> str:
        if not self.history:
            return question
        ctx = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in self.history[-4:])
        return f"Recent conversation:\n{ctx}\n\nUser: {question}"
