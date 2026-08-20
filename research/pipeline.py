"""pipeline.py — the integrated discovery pipeline (spec #9/#11).

Ties the pieces that already exist into one long-horizon assessment loop:

  observe(tool result) -> SANITIZE (trust boundary) -> update the assessment graph + investigation
  state + telemetry -> the model interprets -> add_finding (deduped, evidence-graded HYPOTHESES) ->
  next_step (highest-info-gain action, or escalate on repeated/diminishing) -> ... -> report.

It holds STATE and ORCHESTRATES; it never executes a tool (the session policy + executor do that) and
never authorizes anything. Everything it produces is a hypothesis until a validating test raises it —
and the report renders that honestly. Pure and CPU-testable (drive it with stub results).
"""
from __future__ import annotations

from assessment.graph import AssessmentState
from mcp_layer.telemetry import Telemetry
from research.findings import Hypothesis
from research.investigation import Investigation
from research.report import assessment_report
from trust.boundary import sanitize_untrusted_content


class DiscoveryPipeline:
    def __init__(self, session, telemetry: Telemetry | None = None):
        self.session = session
        self.investigation = Investigation(objective=session.objective)
        self.state = AssessmentState()
        self.telemetry = telemetry or Telemetry(getattr(session, "id", "session"))
        self.findings: list[Hypothesis] = []
        self.timeline: list[str] = []

    # ---- observe a real tool result ---------------------------------------------------------
    def observe(self, tool: str, target: str, arguments: dict, result: dict) -> dict:
        """Route a tool result through the trust boundary, update state + telemetry + investigation.
        Returns the SANITIZED result the model may safely interpret (embedded instructions defanged)."""
        sanitized = sanitize_untrusted_content(f"{tool}:{target}", result)
        try:
            self.state.update_from_tool(tool, arguments or {}, result or {})
        except Exception:                             # noqa: BLE001 — a graph parse hiccup must not stop the loop
            pass
        self.investigation.record_round([{"tool": tool, "arguments": arguments or {},
                                          "result": result or {}}])
        self.telemetry.tool_result(tool, target, (result or {}).get("ok"), result)
        self.timeline.append(f"{tool} {target} -> {'ok' if (result or {}).get('ok') else 'error'}"
                             + ("  [injection defanged]" if sanitized["injection_detected"] else ""))
        return sanitized

    # ---- findings ---------------------------------------------------------------------------
    def add_finding(self, h: Hypothesis) -> bool:
        """Add a finding unless it duplicates one already recorded (same class + component)."""
        key = ((h.vuln_class or "").lower(), (h.affected_component or "").lower())
        if any(((f.vuln_class or "").lower(), (f.affected_component or "").lower()) == key
               for f in self.findings):
            return False
        self.findings.append(h)
        self.investigation.add_hypothesis(h)
        return True

    def confirmed(self) -> list[Hypothesis]:
        return [f for f in self.findings if f.status == "CONFIRMED"]

    # ---- what next --------------------------------------------------------------------------
    def next_step(self) -> dict:
        esc, why = self.investigation.should_escalate()
        if esc:
            return {"action": "escalate", "reason": why}
        nxt = self.investigation.next_action()
        if nxt is None:
            return {"action": "report", "reason": "no open hypotheses — close out and report"}
        return {"action": "test", **nxt}

    # ---- report -----------------------------------------------------------------------------
    def attack_surface(self) -> list[str]:
        try:
            return [ln for ln in self.state.render().splitlines() if ln.strip()][:40]
        except Exception:                             # noqa: BLE001
            return []

    def report(self) -> str:
        rep = assessment_report(objective=self.session.objective, scope=self.session.targets,
                                findings=self.findings, timeline=self.timeline,
                                attack_surface=self.attack_surface())
        self.telemetry.report(f"{len(self.findings)} finding(s), {len(self.confirmed())} confirmed")
        return rep
