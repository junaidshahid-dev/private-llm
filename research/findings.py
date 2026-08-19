"""findings.py — the research reasoning core: structured findings that CANNOT overclaim.

A vulnerability researcher's discipline, made mechanical. The spec's #1 rule is "never treat a
hypothesis as a confirmed vulnerability." Here that is not a request to the model — it is enforced by
the data model: a finding's STATUS is DERIVED from its evidence, never set by hand, and CONFIRMED is
unreachable without a validating test. A model can write "CONFIRMED critical RCE" all it likes;
audit_claim() downgrades it to what the evidence actually supports and says why.

What this gives the rest of the system:
  * Evidence — every point has a source, kind, provenance, confidence (aligns with memory + audit).
  * Hypothesis — the full reasoning schema (observation, why-it-matters, next test, expected result,
    alternative explanation) plus the ranking axes (impact, exploitability, novelty, cost).
  * derive_status — UNCONFIRMED < POSSIBLE < LIKELY < CONFIRMED, from evidence strength; CONFIRMED
    requires a validating test. Nothing else can reach it.
  * rank — orders hypotheses by the value of testing them NEXT (impact × exploitability × how much
    uncertainty a test would remove, minus cost). A CONFIRMED finding has no info gain, so it stops
    competing for the next action — the fix for redundant re-testing.
  * DISCOVERY -> VALIDATION -> EXPLOITATION staging — exploitation is refused until a finding is at
    least LIKELY (validated), and even then it is only a PROPOSAL: execution still needs the human
    approver + an authorized target (the controller enforces that; this module never executes).
  * render_report — the evidence-first report (point 15) with the status label front and centre.

Pure data + pure functions. No model, no network, no execution. Fully unit-tested.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# UNCONFIRMED < POSSIBLE < LIKELY < CONFIRMED — a total order so overclaims are comparable.
STATUS_ORDER = ["UNCONFIRMED", "POSSIBLE", "LIKELY", "CONFIRMED"]
SEVERITY = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
STAGES = ("discovery", "validation", "exploitation")

# how much each kind of evidence can support a claim. ONLY a validating test can reach CONFIRMED.
KIND_WEIGHT = {
    "validated_test": 1.0,   # a (non-destructive) test that actually reached/triggered the condition
    "reproduced": 1.0,       # independently reproduced
    "tool_output": 0.7,      # a real tool result (nmap/http/ffuf/...) — real, but still untrusted data
    "observed": 0.6,         # directly observed behaviour
    "code": 0.6,             # a concrete source path / code location
    "external": 0.4,         # a doc / CVE / advisory reference
    "inference": 0.3,        # model reasoning only, no external support
}
VALIDATING_KINDS = {"validated_test", "reproduced"}
_CERTAINTY = {"UNCONFIRMED": 0.1, "POSSIBLE": 0.4, "LIKELY": 0.75, "CONFIRMED": 1.0}


def _clamp(x, lo=0.0, hi=1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


@dataclass
class Evidence:
    """One piece of support for a hypothesis. Provenance is mandatory — an unsourced 'fact' is not
    evidence. `kind` decides how much weight it can carry (see KIND_WEIGHT)."""
    source: str                       # url / file:path / tool name / memory id — where it came from
    kind: str                         # one of KIND_WEIGHT
    detail: str = ""
    confidence: float = 0.6
    ts: float = field(default_factory=time.time)

    def strength(self) -> float:
        return round(KIND_WEIGHT.get(self.kind, 0.3) * _clamp(self.confidence), 3)

    def is_validating(self) -> bool:
        return self.kind in VALIDATING_KINDS


@dataclass
class Hypothesis:
    """A vulnerability candidate carried as a HYPOTHESIS until evidence earns a higher status."""
    title: str
    vuln_class: str = ""                       # e.g. "SSRF", "IDOR", "path traversal"
    affected_component: str = ""
    observation: str = ""                      # what was seen (point 1: OBSERVATION)
    why_it_matters: str = ""
    next_test: str = ""                        # the safest test that would distinguish it
    expected_result: str = ""
    alternative_explanation: str = ""          # the benign explanation to rule out
    severity: str = "INFO"                     # ASSERTED severity — unvalidated until status earns it
    impact: float = 0.5                        # ranking axes, 0..1
    exploitability: float = 0.5
    novelty: float = 0.3
    cost_to_verify: float = 0.3
    stage: str = "discovery"
    evidence: list[Evidence] = field(default_factory=list)

    # ---- derived, never stored: status/strength/priority always reflect the CURRENT evidence -----
    def evidence_strength(self) -> float:
        return round(min(1.0, sum(e.strength() for e in self.evidence)), 3)

    @property
    def status(self) -> str:
        return derive_status(self.evidence)

    @property
    def confidence_label(self) -> str:
        return self.status

    def priority(self) -> float:
        """Value of testing THIS next: high impact/exploitability + high info gain (uncertainty a
        test would remove) + evidence, minus cost. A CONFIRMED finding has ~0 info gain, so it no
        longer competes for the next action (stops redundant re-testing)."""
        info_gain = 1.0 - _CERTAINTY[self.status]
        score = (0.30 * _clamp(self.impact) + 0.25 * _clamp(self.exploitability)
                 + 0.20 * info_gain + 0.15 * _clamp(self.novelty)
                 + 0.10 * self.evidence_strength() - 0.20 * _clamp(self.cost_to_verify))
        return round(_clamp(score), 3)


def derive_status(evidence: list[Evidence]) -> str:
    """UNCONFIRMED < POSSIBLE < LIKELY < CONFIRMED, from evidence alone. CONFIRMED is reachable ONLY
    with a validating test — this is the mechanical guarantee that a hypothesis never masquerades as
    a confirmed vulnerability."""
    if any(e.is_validating() for e in evidence):
        return "CONFIRMED"
    strength = min(1.0, sum(e.strength() for e in evidence))
    if strength >= 0.6:
        return "LIKELY"
    if strength >= 0.15:
        return "POSSIBLE"
    return "UNCONFIRMED"


def audit_claim(claimed_status: str, evidence: list[Evidence]) -> tuple[bool, str, str]:
    """Compare a CLAIMED status against what the evidence supports. Returns (ok, actual, reason).
    The overclaim guard: if the claim outranks the evidence, it is downgraded and flagged."""
    claimed = (claimed_status or "").upper()
    actual = derive_status(evidence)
    if claimed not in STATUS_ORDER:
        return False, actual, f"unknown status {claimed_status!r}; evidence supports {actual}"
    if STATUS_ORDER.index(claimed) > STATUS_ORDER.index(actual):
        need = "a validating (non-destructive) test" if actual != "CONFIRMED" else "more evidence"
        return False, actual, (f"OVERCLAIM: '{claimed}' asserted but evidence supports only "
                               f"'{actual}' — needs {need}. Downgraded.")
    return True, actual, f"claim '{claimed}' is supported (evidence -> {actual})"


def rank(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Highest value-to-test-next first. Deterministic (title breaks ties)."""
    return sorted(hypotheses, key=lambda h: (-h.priority(), h.title))


def exploitation_gate(h: Hypothesis) -> tuple[bool, str]:
    """DISCOVERY -> VALIDATION -> EXPLOITATION. Exploitation is refused until the finding is at least
    LIKELY (validated non-destructively). Even then this only PERMITS a proposal — execution still
    requires the human approver + an authorized target, enforced by the controller, not here."""
    if h.status in ("LIKELY", "CONFIRMED"):
        return True, ("validated enough to PROPOSE exploitation — but execution requires explicit "
                      "human approval and an authorized target (controller-enforced). Prefer a "
                      "minimal, non-destructive proof of concept.")
    return False, (f"status is {h.status}: validate non-destructively first (prove input reaches the "
                   "sink) before proposing anything that could modify or damage the target.")


def render_reasoning(h: Hypothesis) -> str:
    """The point-1 structured reasoning block for a single hypothesis."""
    L = [f"OBSERVATION: {h.observation or '—'}",
         "EVIDENCE:"]
    for e in h.evidence:
        L.append(f"  - [{e.kind}] {e.detail or e.source} (source={e.source}, conf={e.confidence})")
    if not h.evidence:
        L.append("  - (none yet)")
    L += [f"HYPOTHESIS: {h.title}" + (f" [{h.vuln_class}]" if h.vuln_class else ""),
          f"WHY IT MATTERS: {h.why_it_matters or '—'}",
          f"CONFIDENCE: {h.status} (evidence strength {h.evidence_strength()})",
          f"NEXT TEST: {h.next_test or '—'}",
          f"EXPECTED RESULT: {h.expected_result or '—'}",
          f"ALTERNATIVE EXPLANATION: {h.alternative_explanation or '—'}"]
    return "\n".join(L)


def render_report(h: Hypothesis) -> str:
    """The point-15 evidence-first vulnerability report. Status is front and centre; severity is
    labelled as ASSERTED until the status earns it (LIKELY/CONFIRMED)."""
    validated = h.status in ("LIKELY", "CONFIRMED")
    sev = f"{h.severity}" + ("" if validated else "  (ASSERTED — unvalidated; treat as a hypothesis)")
    L = [f"# {h.title}",
         f"STATUS: {h.status}    SEVERITY: {sev}    CONFIDENCE: {h.status}",
         f"Vulnerability class: {h.vuln_class or '—'}",
         f"Affected component: {h.affected_component or '—'}",
         "",
         "## Evidence"]
    for e in h.evidence:
        L.append(f"- [{e.kind}, conf {e.confidence}] {e.detail or ''} (source: {e.source})")
    if not h.evidence:
        L.append("- (no evidence gathered — UNCONFIRMED)")
    L += ["",
          f"## Technical explanation\n{h.why_it_matters or '—'}",
          f"\n## Attack preconditions / limitations\n{h.alternative_explanation or 'stated as reasoning; validate before relying'}",
          f"\n## Reproduction / next test\nExpected behaviour: {h.expected_result or '—'}",
          f"Observed behaviour: {h.observation or '—'}",
          f"Next test to run (needs authorization if active): {h.next_test or '—'}",
          "",
          f"## Recommended remediation\n{_remediation_hint(h.vuln_class)}"]
    if not validated:
        L.append("\n> This is a HYPOTHESIS, not a confirmed vulnerability. Do not report it as "
                 "confirmed until a validating test raises the status to LIKELY/CONFIRMED.")
    return "\n".join(L)


def _remediation_hint(vuln_class: str) -> str:
    return {"sqli": "Use parameterized queries.", "xss": "Context-aware output encoding + CSP.",
            "ssrf": "Allowlist egress; block internal/link-local ranges.",
            "idor": "Enforce server-side object-level authorization per request.",
            "path traversal": "Canonicalize and confine paths under the base directory."
            }.get((vuln_class or "").lower(), "Remediate the root cause; verify the fix.")
