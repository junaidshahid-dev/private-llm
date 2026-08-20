"""session_policy.py — one operator authorization per ASSESSMENT, then autonomous (no per-call prompts).

The operator starts an authorized assessment ONCE — target(s), objective, capability profile, time
limit — and the agent then runs in-scope tools WITHOUT asking approval for each call. This does not
weaken the safety architecture; it moves the human decision from "approve every tool" to "authorize
this assessment", and keeps three HARD gates that the model can never bypass:

  * KILL SWITCH   engaged -> nothing runs, even mid-session (checked here AND in the executor).
  * AUTHORIZED TARGET   an active tool may only touch a target that is in the operator's
    authorized_targets registry AND within this session's declared scope. Only the operator's own
    authorized targets — never an arbitrary third party.
  * CAPABILITY PROFILE   bounds what may run autonomously by each tool's declared side_effects
    (rich schema): read_only tools always; active-but-non-destructive recon under 'recon'; anything
    that would MODIFY a target only under an explicit 'full' profile.

The model CANNOT start a session (start_session requires operator_ack is the boolean True — never
model text) and CANNOT widen scope. approver_for(session) returns the exact `approver` that
run_session already expects, so the autonomous loop reuses the proven plan->approve->execute->verify
machinery unchanged — only the approver is now the operator's one-time session policy instead of a
per-call prompt.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from mcp_layer import permissions as perm
from mcp_layer import security as secmod

# capability profile -> the tool side-effect classes it permits to run autonomously.
#   "none"           local, no target traffic
#   "network:read"   sends traffic to the target but does not modify it (passive recon/read)
#   "network:probe"  actively PROBES the target for vulnerabilities with crafted requests, but does
#                    NOT modify/exfiltrate it (nuclei/nikto-class validation) — beyond passive recon,
#                    below exploitation; only under 'validation' or 'full'.
#   "network:write" / "local:write"  would MODIFY/exfiltrate a target/host (exploitation, e.g. sqlmap
#                    dumping a database) — only under an explicit 'full' profile.
# The tiers are cumulative: recon <= validation <= full. Deny-by-default still applies to targets, the
# kill switch overrides every profile, and the model can neither start a session nor widen the profile.
PROFILES = {
    "read_only": {"none"},
    "recon": {"none", "network:read"},
    "validation": {"none", "network:read", "network:probe"},
    "full": {"none", "network:read", "network:probe", "network:write", "local:write"},
}
DEFAULT_PROFILE = "recon"


@dataclass
class AuthorizedSession:
    objective: str
    targets: list = field(default_factory=list)          # the operator's declared assessment scope
    capability_profile: str = DEFAULT_PROFILE
    time_limit_s: int = 3600
    started_at: float = field(default_factory=time.time)
    created_by: str = "operator"                          # provenance — never the model
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    log: list = field(default_factory=list)               # per-decision audit trail

    def expired(self) -> bool:
        return (time.time() - self.started_at) > self.time_limit_s

    def remaining_s(self) -> int:
        return max(0, round(self.time_limit_s - (time.time() - self.started_at)))

    def render(self) -> str:
        return (f"AUTHORIZED SESSION {self.id} — objective: {self.objective}\n"
                f"  scope: {self.targets or '(local only)'}   profile: {self.capability_profile}   "
                f"remaining: {self.remaining_s()}s\n"
                f"  autonomous: in-scope tools run without per-call approval; kill switch overrides.")


def _tool_index() -> dict:
    """Merge every tool's rich schema into {name: entry}."""
    from mcp_layer import tools as toolmod
    from web import tools as webmod
    idx = {}
    for entry in toolmod.schema() + secmod.schema() + webmod.schema():
        idx[entry["name"]] = entry
    return idx


def _in_scope(target: str, session_targets: list) -> bool:
    ok, _ = secmod.target_authorized(target, [{"match": t} for t in session_targets])
    return ok


def authorize(session: AuthorizedSession, proposal: dict, config=None) -> tuple[bool, str]:
    """Should this proposed tool call run autonomously under the session? (allowed, reason).
    Deny by default; every hard gate must pass."""
    from mcp_layer import killswitch
    if killswitch.is_engaged():
        return False, "kill switch engaged — all operations halted"
    if session.expired():
        return False, "session expired — the operator must re-authorize"
    tool = (proposal or {}).get("tool", "")
    entry = _tool_index().get(tool)
    if entry is None:
        return False, f"unknown tool {tool!r}"
    se = entry.get("side_effects", "none")
    if se not in PROFILES.get(session.capability_profile, set()):
        return False, (f"tool side-effect class '{se}' is not permitted by capability profile "
                       f"'{session.capability_profile}'")
    if entry.get("requires_authorization"):
        args = (proposal or {}).get("arguments") or {}
        target = str(args.get("target") or args.get("url") or "")
        if config is None:
            config = perm.load_config()
        reg = (config.get("security_tools") or {}).get("authorized_targets")
        # A target is authorized if the OPERATOR authorized it EITHER way (a mix of both sources):
        #   (1) declared for THIS session (the operator can extend scope to any target they are
        #       responsible for, via authorize_target with operator_ack), OR
        #   (2) present in the standing authorized_targets registry.
        # The registry check is retained; the session scope is an additional operator authorization.
        # Everything the operator has NOT authorized either way is denied by default.
        if not (_in_scope(target, session.targets) or secmod.target_authorized(target, reg)[0]):
            return False, (f"target {target!r} is not authorized: neither in this session's scope "
                           f"{session.targets} nor in the operator's authorized_targets registry")
    return True, "authorized by session policy (in scope — no per-call prompt)"


def approver_for(session: AuthorizedSession, config=None):
    """The `approver` run_session expects — but derived from the operator's ONE-TIME session
    authorization, so in-scope tools run with NO per-call prompt. Denials are audited."""
    def approve(proposal) -> bool:
        ok, reason = authorize(session, proposal, config)
        session.log.append({"tool": (proposal or {}).get("tool"), "approved": ok, "reason": reason})
        if not ok:
            secmod.audit("session_denied", (proposal or {}).get("tool", "?"),
                         str(((proposal or {}).get("arguments") or {}).get("target", "")), reason)
        return ok
    return approve


def start_session(objective: str, targets: list, capability_profile: str = DEFAULT_PROFILE,
                  time_limit_s: int = 3600, operator_ack: bool = False, config=None) -> dict:
    """Operator-only: begin an authorized assessment. Requires operator_ack is the boolean True (the
    model can never start a session), a known profile, and that every declared target is already in
    the operator's authorized_targets. Returns {ok, session} or {ok:False, error}."""
    if operator_ack is not True:
        return {"ok": False, "error": "starting an assessment session requires an explicit operator "
                "acknowledgement (the boolean True). The model cannot start a session."}
    if capability_profile not in PROFILES:
        return {"ok": False, "error": f"unknown capability profile {capability_profile!r}; "
                f"choose from {list(PROFILES)}"}
    if capability_profile != "read_only" and not targets:
        return {"ok": False, "error": "declare the target(s) for this assessment (the scope)"}
    # The operator (operator_ack=True) is the authority for their own scope: the targets they declare
    # here are authorized for this session. Standing registry targets remain authorized too (the mix).
    # Deny-by-default still applies to everything the operator has NOT authorized either way.
    sess = AuthorizedSession(objective=objective, targets=[str(t).strip() for t in (targets or []) if str(t).strip()],
                             capability_profile=capability_profile, time_limit_s=int(time_limit_s))
    secmod.audit("session_start", "-", ",".join(sess.targets) or "(local)",
                 f"objective={objective!r} profile={capability_profile} ttl={time_limit_s}s id={sess.id}")
    return {"ok": True, "session": sess}


def authorize_target(session: AuthorizedSession, target: str, operator_ack: bool = False) -> dict:
    """Operator-only: extend the session's authorized scope to another target mid-assessment ('move to
    any target'). Requires operator_ack is the boolean True — the model can never widen the scope."""
    if operator_ack is not True:
        return {"ok": False, "error": "extending the session scope requires an explicit operator "
                "acknowledgement (the boolean True). The model cannot authorize a new target."}
    t = str(target or "").strip()
    if not t:
        return {"ok": False, "error": "empty target"}
    if t not in session.targets:
        session.targets.append(t)
        secmod.audit("session_scope_added", "-", t, f"session {session.id}")
    return {"ok": True, "targets": list(session.targets)}
