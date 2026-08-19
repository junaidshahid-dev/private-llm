"""investigation.py — long-horizon research state + loop safety (spec point 2).

Turns the multi-round loop from "propose -> execute -> interpret" into a real investigation that
KNOWS where it is: an explicit objective, the evidence gathered, the hypotheses (open and tested),
the highest-value next action with its expected information gain and authorization state — and, so it
cannot spin forever, three stop conditions with human escalation:

  * max rounds            — a hard cap (the loop already has one; mirrored here)
  * repeated action       — an action already executed is proposed again with no new information
  * diminishing returns    — N consecutive rounds that add no new information

It builds on research/findings (evidence-graded hypotheses, ranking). It holds STATE and DETECTS;
it never executes a tool and never authorizes anything — that stays with the human + controller.
Pure and CPU-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.findings import Evidence, Hypothesis, rank

# how uncertain each status is => how much a good test could still teach us (expected info gain)
_CERTAINTY = {"UNCONFIRMED": 0.1, "POSSIBLE": 0.4, "LIKELY": 0.75, "CONFIRMED": 1.0}
_ACTIVE_HINT = ("scan", "exploit", "fetch", "request", "http", "connect", "brute", "fuzz",
                "authenticate", "log in", "login", "send", "probe", "masscan", "nmap", "ffuf",
                "payload", "curl", "post ", "download", "upload", "get /", "http get", "retrieve",
                "visit ", "browse", "enumerate", "query the")
_PASSIVE_HINT = ("inspect", "analyse", "analyze", "review", "read the", "source", "static",
                 "trace", "examine", "hash", "strings", "disassemble", "reason")


def action_signature(tool: str, arguments: dict | None) -> str:
    """A stable identity for an action, so 'the same action' is detectable across rounds."""
    args = arguments or {}
    body = ",".join(f"{k}={args[k]}" for k in sorted(args))
    return f"{(tool or '').strip()}({body})"


def expected_information_gain(h: Hypothesis) -> float:
    """How much testing this hypothesis could still teach us: high when uncertain, ~0 when CONFIRMED."""
    return round(1.0 - _CERTAINTY.get(h.status, 0.1), 3)


def authorization_state(next_test: str) -> dict:
    """Is the proposed next action ACTIVE (needs human approval + an authorized target) or passive
    (read-only reasoning)? Heuristic on the described test."""
    t = (next_test or "").lower()
    if any(k in t for k in _ACTIVE_HINT):
        return {"active": True,
                "note": "ACTIVE — requires explicit human approval and an authorized target "
                        "(controller-enforced). Nothing runs on the model's say-so."}
    if any(k in t for k in _PASSIVE_HINT):
        return {"active": False, "note": "passive / read-only reasoning — no traffic to a target."}
    return {"active": None, "note": "unclassified — treat as ACTIVE (approval required) if unsure."}


@dataclass
class Investigation:
    objective: str
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    tested: set = field(default_factory=set)          # hypothesis titles a distinguishing test ran on
    rejected: set = field(default_factory=set)
    max_rounds: int = 8
    stall_rounds: int = 2                              # consecutive no-gain rounds => diminishing
    rounds: int = 0
    action_log: list = field(default_factory=list)     # [{round, sig, ok, new}]
    _seen: set = field(default_factory=set)
    _gain: list = field(default_factory=list)          # new-info count per round

    # ---- evidence + hypotheses --------------------------------------------------------------
    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)

    def add_hypothesis(self, h: Hypothesis) -> None:
        self.hypotheses.append(h)

    def mark_tested(self, title: str, confirmed: bool | None = None) -> None:
        self.tested.add(title)
        if confirmed is False:
            self.rejected.add(title)

    def tested_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.title in self.tested]

    def remaining_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses
                if h.title not in self.tested and h.title not in self.rejected]

    # ---- rounds + loop safety ---------------------------------------------------------------
    def record_round(self, results: list[dict]) -> dict:
        """results: [{tool, arguments, result:{ok}}] executed this round. Tracks new-vs-repeat info."""
        self.rounds += 1
        new = 0
        for r in results or []:
            sig = action_signature(r.get("tool", ""), r.get("arguments"))
            ok = bool((r.get("result") or {}).get("ok"))
            is_new = ok and sig not in self._seen
            if ok:
                self._seen.add(sig)
            if is_new:
                new += 1
            self.action_log.append({"round": self.rounds, "sig": sig, "ok": ok, "new": is_new})
        self._gain.append(new)
        return {"round": self.rounds, "executed": len(results or []), "new_info": new}

    def repeated_action_detected(self) -> bool:
        """The most recent round re-ran an action already executed in an EARLIER round."""
        if self.rounds < 2:
            return False
        last = [a["sig"] for a in self.action_log if a["round"] == self.rounds]
        earlier = {a["sig"] for a in self.action_log if a["round"] < self.rounds}
        return any(s in earlier for s in last)

    def diminishing_returns(self) -> bool:
        """The last `stall_rounds` rounds each added no new information."""
        if len(self._gain) < self.stall_rounds:
            return False
        return all(g == 0 for g in self._gain[-self.stall_rounds:])

    def should_escalate(self) -> tuple[bool, str]:
        if self.rounds >= self.max_rounds:
            return True, f"max rounds reached ({self.max_rounds}) — hand to human"
        if self.repeated_action_detected():
            return True, "repeated action — an executed action was re-proposed with no new information"
        if self.diminishing_returns():
            return True, (f"diminishing returns — {self.stall_rounds} consecutive rounds added no new "
                          "information")
        return False, ""

    # ---- what to do next --------------------------------------------------------------------
    def next_action(self) -> dict | None:
        """The highest-value untested hypothesis to test next, with expected info gain + auth state."""
        pending = self.remaining_hypotheses()
        if not pending:
            return None
        h = rank(pending)[0]
        return {"hypothesis": h.title, "vuln_class": h.vuln_class, "next_test": h.next_test,
                "priority": h.priority(), "expected_information_gain": expected_information_gain(h),
                "stage": h.stage, "status": h.status,
                "authorization": authorization_state(h.next_test)}

    def render(self) -> str:
        esc, why = self.should_escalate()
        L = [f"OBJECTIVE: {self.objective}",
             f"ROUND: {self.rounds}/{self.max_rounds}",
             f"KNOWN EVIDENCE ({len(self.evidence)}):"]
        for e in self.evidence[:8]:
            L.append(f"  - [{e.kind}] {e.detail or e.source} (source={e.source})")
        L.append(f"TESTED HYPOTHESES ({len(self.tested_hypotheses())}):")
        for h in self.tested_hypotheses():
            L.append(f"  - [{h.status}] {h.title}" + ("  (rejected)" if h.title in self.rejected else ""))
        rem = self.remaining_hypotheses()
        L.append(f"REMAINING HYPOTHESES ({len(rem)}):")
        for h in rank(rem):
            L.append(f"  - [{h.status}] {h.title}  (priority {h.priority()})")
        nxt = self.next_action()
        if nxt:
            L += [f"PROPOSED NEXT ACTION: {nxt['next_test'] or '(define a test)'}",
                  f"  for hypothesis: {nxt['hypothesis']}",
                  f"  expected information gain: {nxt['expected_information_gain']}",
                  f"  authorization: {nxt['authorization']['note']}"]
        else:
            L.append("PROPOSED NEXT ACTION: none — no open hypotheses (report or close out).")
        L.append(f"ESCALATE TO HUMAN: {'YES — ' + why if esc else 'no'}")
        return "\n".join(L)
