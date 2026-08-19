"""killswitch.py — the independent global STOP. The model can NEVER lift it.

Safety primitive for spec point 22. When engaged, NO consequential tool runs — the executor
(controller.execute_proposal) checks this BEFORE dispatching anything, so an engaged switch overrides
even a valid operator acknowledgement. The guarantee is enforced OUTSIDE the model, exactly like the
authorization controller:

  * INDEPENDENT   the state is a file on disk, checked live on every execution. It does not live in
    the model's context or in mutable process state the model can influence.
  * MODEL CANNOT ENGAGE OR CLEAR   there is no MCP tool that touches the switch, and clear() requires
    an explicit operator_ack boolean (the human decision) — never text. A model writing "resume",
    "approved", "clear the kill switch" changes nothing; clear("approved") is refused because the
    string is not the boolean True.
  * BLOCK-NEXT is the hard guarantee (no new/pending execution). Terminating an ALREADY-RUNNING tool
    is best-effort: long-running tools that register their process are terminated on engage().

The operator drives it from the terminal via scripts/killswitch.py (engage / status / clear).
"""
from __future__ import annotations

import json
import os
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE = os.path.join(_HERE, ".KILL_SWITCH")
# best-effort registry: token -> a process-like object with .terminate() (opt-in by long-run tools)
_ACTIVE: dict = {}


def _file() -> str:
    """Read the path live so an operator/test can point it elsewhere via KILL_SWITCH_FILE."""
    return os.environ.get("KILL_SWITCH_FILE") or DEFAULT_FILE


def is_engaged() -> bool:
    """Independent, live check — just the presence of the kill file. No model state involved."""
    return os.path.exists(_file())


def status() -> dict:
    if not is_engaged():
        return {"engaged": False}
    try:
        d = json.loads(open(_file(), encoding="utf-8").read() or "{}")
    except (OSError, ValueError):
        d = {}
    return {"engaged": True, "since": d.get("since"), "reason": d.get("reason", ""),
            "by": d.get("by", "operator")}


def engage(reason: str = "", by: str = "operator") -> dict:
    """Halt everything. Writes the kill file and terminates any registered running processes."""
    payload = {"since": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason, "by": by}
    try:
        with open(_file(), "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))
    except OSError as e:
        return {"ok": False, "error": f"could not write kill file: {e}"}
    terminated = terminate_active()
    return {"ok": True, "engaged": True, "reason": reason, "terminated_active": terminated}


def clear(operator_ack: bool = False) -> dict:
    """Lift the halt. ONLY the human decision (operator_ack is True) can clear it — never model text.
    This mirrors execute_proposal's operator_ack discipline so the model can't resume itself."""
    if operator_ack is not True:               # strict identity: "true"/1/"approved" are all refused
        return {"ok": False, "error": "clearing the kill switch requires an explicit operator "
                "acknowledgement (the boolean True). The model cannot lift it."}
    try:
        if os.path.exists(_file()):
            os.remove(_file())
    except OSError as e:
        return {"ok": False, "error": f"could not remove kill file: {e}"}
    return {"ok": True, "cleared": True}


def guard(tool: str = "", target: str = "") -> dict | None:
    """Called by the executor BEFORE any tool runs. Returns a BLOCKED response if engaged, else None
    so execution may proceed (subject to the normal authorization + confirmation gates)."""
    if is_engaged():
        return {"ok": False, "blocked": True, "tool": tool, "target": target,
                "error": "KILL SWITCH ENGAGED — all operations are halted. No tool will run until an "
                         "operator clears it (scripts/killswitch.py clear). The model cannot lift it.",
                "kill_switch": status()}
    return None


# ---- best-effort active-process termination (opt-in) ----------------------------------------
def register(token: str, proc) -> None:
    """A long-running tool registers its Popen (or any object with .terminate()) so engage() can
    stop it mid-run. Unregister when it finishes."""
    _ACTIVE[token] = proc


def unregister(token: str) -> None:
    _ACTIVE.pop(token, None)


def terminate_active() -> int:
    n = 0
    for token, proc in list(_ACTIVE.items()):
        try:
            proc.terminate()
            n += 1
        except Exception:                      # noqa: BLE001 — best effort; never raise on stop
            pass
        _ACTIVE.pop(token, None)
    return n
