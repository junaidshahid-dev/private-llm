"""verify.py — the verification layer (north-star principle 4).

    from verification.verify import verify
    findings = verify(answer_text, hits=retrieved_hits, tools_ran=[...])

Runs AFTER generation, BEFORE the assistant asserts an answer. It does not rewrite the
answer; it returns FINDINGS the caller can surface ("I computed 9.8, but wrote 0.098 —
flagging that") so a mistake is DETECTED instead of confidently hidden. Today this was done by
hand: a security-benchmark regression (network 1.00->0.50) hid inside a rising average and only a
human reading the per-domain table caught it. This module is that human, made automatic and
general.

Design rules, learned the hard way:
  * A verifier that cries wolf is worse than none — you stop reading it. So each check is
    CONSERVATIVE: it only fires on things it can actually decide. Prose full of numbers
    (ports, CVE ids, versions, dates) must NOT trip the math check.
  * Deterministic checks (math, code) are ERROR-level: a mismatch is a real defect.
    Heuristic checks (grounding, phantom-action) are WARN/INFO: signals, not verdicts, and they
    say so.
  * Every check states what it can and cannot prove. No check claims more certainty than it has —
    that would be the very failure mode this layer exists to catch.

Pure-CPU, no model, no network. Unit-tested by verification/verify_test.py with planted errors:
if it cannot catch a deliberately broken answer, it cannot be trusted on a real one.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field


# ---- finding ----------------------------------------------------------------
@dataclass
class Finding:
    level: str          # "error" (deterministic defect) | "warn" | "info" (heuristic signal)
    kind: str           # math|code|grounding|phantom_action|tool_grounding|cve_format|
                        # claim_grounding|overclaim|severity|target_authorization
    detail: str
    span: str = ""      # the offending text, for the caller to quote back

    def __str__(self) -> str:
        s = f"[{self.level.upper()}:{self.kind}] {self.detail}"
        return s + (f'  ->  "{self.span}"' if self.span else "")


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    # The verifier is NON-DESTRUCTIVE: it never edits the answer, it classifies it. A three-tier
    # verdict, plus the recommended next move, so the SYSTEM decides what to do (regenerate,
    # retrieve more, run a tool, or present the uncertainty) — the verifier does not act on its own,
    # which is what keeps a heuristic from becoming a second hallucination source.
    _NEXT = {
        "PASS":    "present the answer (no checkable defect found).",
        "WARNING": "surface the flag; consider retrieving more evidence, running a tool, or "
                   "presenting the point as unverified before relying on it.",
        "BLOCK":   "do NOT present as trusted; regenerate, retrieve, run the tool, or hand back "
                   "for human review.",
    }

    @property
    def verdict(self) -> str:
        if any(f.level == "error" for f in self.findings):
            return "BLOCK"
        if any(f.level == "warn" for f in self.findings):
            return "WARNING"
        return "PASS"

    @property
    def ok(self) -> bool:
        return self.verdict != "BLOCK"

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    def add(self, *fs: Finding) -> None:
        self.findings.extend(fs)

    def render(self) -> str:
        """Human-facing block. States the verdict, the findings, and the recommended next move —
        and never claims that a clean pass proves correctness."""
        v = self.verdict
        lines = [f"Verification: {v}"]
        if not self.findings:
            lines.append("  no checkable issue found "
                         "(absence of a flag is not proof of correctness).")
        for f in self.findings:
            lines.append("  - " + str(f))
        lines.append("  next: " + self._NEXT[v])
        return "\n".join(lines)

    # kept for the CLI / older callers; render() is the richer form
    def summary(self) -> str:
        return self.render()


# ---- 1. math: recompute arithmetic the answer states ------------------------
# Only PURE arithmetic is evaluated. A number that is not part of an "expr = result" (or a
# percentage form) is left alone — that is how ports, CVE-2007-2447, 3.0.20 and 2026-08-15 avoid
# being "checked". eval() is never used; a whitelisted-AST evaluator handles +-*/ and parentheses.
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos, ast.Mod: operator.mod}


def _safe_arith(expr: str):
    """Evaluate a pure-arithmetic string, or return None if it is not pure arithmetic."""
    expr = expr.strip().rstrip("%").replace(",", "").replace("×", "*").replace("÷", "/")
    if not expr or not re.fullmatch(r"[\d\.\s()+\-*/%^]+", expr):
        return None
    expr = expr.replace("^", "**")
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            if type(n.op) is ast.Div and b == 0:
                return None
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            v = ev(n.operand)
            return None if v is None else _OPS[type(n.op)](v)
        return None
    try:
        return ev(node)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-6, abs(b) * 1e-3)   # 0.1% relative, so rounded results pass


def check_math(answer: str) -> list[Finding]:
    out: list[Finding] = []
    seen = set()

    # form A: "<arith expression> = <result>"  (both sides present in the text)
    for m in re.finditer(r"([\d,][\d,.\s()+\-*/%×÷^]*?)\s*=\s*(-?[\d,]+(?:\.\d+)?)\s*(%?)", answer):
        lhs, rhs_s, pct = m.group(1), m.group(2), m.group(3)
        if not re.search(r"[+\-*/×÷^]", lhs):     # no operator => not a computation, skip
            continue
        val = _safe_arith(lhs)
        if val is None:
            continue
        rhs = _safe_arith(rhs_s)
        if rhs is None:
            continue
        key = m.group(0).strip()
        if key in seen:
            continue
        seen.add(key)
        if not _close(val, rhs):
            out.append(Finding("error", "math",
                               f"stated {lhs.strip()} = {rhs_s}{pct}, but it computes to "
                               f"{_fmt(val)}", key))

    # form B: "<p>% of <n>" followed by "= <result>" or "is <result>"
    for m in re.finditer(r"(-?[\d.]+)\s*%\s*of\s*(-?[\d,]+(?:\.\d+)?)\s*(?:=|is|equals)\s*"
                         r"(-?[\d,]+(?:\.\d+)?)", answer, re.I):
        p, n, r = (_safe_arith(x) for x in m.groups())
        if None in (p, n, r):
            continue
        want = p / 100.0 * n
        if not _close(want, r):
            out.append(Finding("error", "math",
                               f"{m.group(1)}% of {m.group(2)} is {_fmt(want)}, not {m.group(3)}",
                               m.group(0)))
    return out


def _fmt(x: float) -> str:
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.6g}"


# ---- 2. code: syntax-check (and run if self-testing) ------------------------
# Only ```python / ```py FENCED blocks are checked. grading.py's _extract_code has a benchmark
# fallback that returns bare prose as "code" (there the whole answer is expected to be a function);
# reusing it here made the verifier compile plain sentences and cry "syntax error" on "CVE-2007-
# 2447". Conservative wins: an unlabelled or shell fence is skipped (miss) rather than mis-flagged.
def _fenced_python(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text or "", re.S)
    return "\n\n".join(b for b in blocks).strip()


def check_code(answer: str) -> list[Finding]:
    code = _fenced_python(answer)
    if not code:
        return []
    try:
        compile(code, "<answer>", "exec")
    except SyntaxError as e:
        return [Finding("error", "code", f"code block does not parse: {e.msg} (line {e.lineno})",
                        code[:80])]
    return [Finding("info", "code",
                    "code parses; not executed (no self-test present). Syntax-valid is not "
                    "correctness — run it against a test to verify behaviour.")]


# ---- 3. grounding: are the answer's specifics supported by the context? -----
# HEURISTIC and WARN-level, deliberately. It cannot understand meaning; it checks whether the
# concrete tokens an answer leans on (cited [n], numbers, ALLCAPS/CamelCase identifiers) actually
# appear in the retrieved context. A miss is a PROMPT TO CHECK, not a proven hallucination — the
# fact may be correct parametric knowledge the context simply doesn't cover (which the grounding
# CONTRACT now explicitly allows). It exists to make "cite [n]" auditable, not to gag the model.
_STOP = {"HTTP", "HTTPS", "HTML", "JSON", "SQL", "API", "URL", "TLS", "SSL", "TCP", "UDP", "OS",
         "ID", "OK", "GET", "POST", "IP", "DNS", "AES", "RSA", "MD5", "SHA", "JWT", "XSS", "CSRF",
         "SSRF", "IDOR", "RCE", "VM", "AND", "OR", "NOT", "IMDS", "IAM"}


def check_grounding(answer: str, hits: list[dict] | None) -> list[Finding]:
    if not hits:
        return []
    n_sources = len(hits)
    ctx = " ".join(h.get("text", "") for h in hits).lower()
    out: list[Finding] = []

    # cited [n] must point at a source that exists
    for cit in sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)}):
        if cit == 0 or cit > n_sources:
            out.append(Finding("warn", "grounding",
                               f"answer cites [{cit}] but only {n_sources} sources were retrieved",
                               f"[{cit}]"))

    # A specific FACTUAL claim absent from the sources (a CVE id) is a WARNING — the system should
    # verify it before relying on it. It is NOT an error: after the grounding-contract fix the model
    # may legitimately supply correct parametric knowledge the context does not cover. The verifier
    # surfaces; it does not overrule.
    cves = sorted({c for c in re.findall(r"\bCVE-\d{4}-\d{3,7}\b", answer) if c.lower() not in ctx})
    if cves:
        out.append(Finding("warn", "grounding",
                           "factual claim not found in the retrieved sources (verify before relying "
                           "- may be correct knowledge, may be invented): " + ", ".join(cves)))

    # Other identifiers (CamelCase, ALLCAPS) absent from context are lower signal -> INFO only, so
    # warnings stay meaningful rather than firing on every capitalised word.
    other = {t for t in re.findall(r"\b[A-Z][A-Za-z]*(?:[A-Z][A-Za-z]+)+\b", answer)}   # CamelCase
    other |= {t for t in re.findall(r"\b[A-Z]{3,}\b", answer) if t not in _STOP}
    other = sorted({t for t in other if t.lower() not in ctx and not t.startswith("CVE")})
    if other:
        out.append(Finding("info", "grounding",
                           "other specific terms not in the retrieved context (lower signal, worth "
                           "a glance): " + ", ".join(other[:8])))
    return out


# ---- 4. phantom actions: did the answer claim to DO something no tool did? ---
# The reasoning/execution boundary in prose. controller.py stops the model from EXECUTING; this
# stops it from CLAIMING it executed. If the text says "I scanned/ran/accessed ..." and no tool
# actually ran, that is a fabricated action — an error, because it misrepresents what happened.
_ACTION = re.compile(
    r"\bI\s+(?:have\s+)?(?:just\s+)?(scanned|ran|executed|accessed|connected|downloaded|"
    r"fetched|queried|opened|deleted|modified|wrote|sent|installed|compiled|logged\s+in|"
    r"nmapped|exploited|retrieved)\b", re.I)


def check_phantom_actions(answer: str, tools_ran: list[str] | None) -> list[Finding]:
    if tools_ran:                       # a tool really ran; "I ran ..." may be truthful
        return []
    out = []
    for m in _ACTION.finditer(answer):
        s, e = max(0, m.start() - 10), min(len(answer), m.end() + 40)
        out.append(Finding("error", "phantom_action",
                           f"answer claims to have performed an action ('{m.group(0)}') but no "
                           "tool was executed - it may only describe, not assert it did it",
                           answer[s:e].strip()))
    return out


# ---- 5. tool grounding: did the answer FABRICATE tool output? ---------------
# The most dangerous failure, and the one a live run exposed: every tool call errored (no readable
# root on Kaggle), the model saw only errors, then presented a fabricated git-status block and a
# fake revision "a1b2c3d4" as if real — and every other check passed it. phantom_action missed it
# because the model never wrote "I ran ..."; it just PRESENTED fake output. This check closes that:
# if tools were attempted and EVERY one errored, the answer must ACKNOWLEDGE the failure, not report
# results. Narrow on purpose (all-errored, none-succeeded) so it fires on fabrication, not on a
# normal answer that mixes real tool output with reasoning.
_FAILURE_ACK = re.compile(
    r"\b(error|errored|failed|failure|could not|couldn'?t|cannot|can'?t|unable|not found|"
    r"does not exist|doesn'?t exist|no such|not a file|not a directory|not accessible|"
    r"outside the allowed|permission denied|denied|no results?|returned an error|"
    r"did not (?:return|work)|was not able)\b", re.I)


def check_tool_grounding(answer: str, tool_results: list[dict] | None) -> list[Finding]:
    if not tool_results:
        return []
    oks = [r for r in tool_results if (r.get("result") or {}).get("ok")]
    errs = [r for r in tool_results if not (r.get("result") or {}).get("ok")]
    if oks or not errs:
        return []                      # something succeeded, or nothing errored -> not this case
    if _FAILURE_ACK.search(answer or ""):
        return []                      # honestly reports the failure -> fine
    return [Finding("error", "tool_grounding",
                   f"all {len(errs)} tool call(s) errored, yet the answer presents results without "
                   "acknowledging the failure - it appears to FABRICATE tool output",
                   (answer or "").strip()[:120])]


# ---- 6. security specifics: CVE ids and hashes ------------------------------
# Deterministic, security-relevant, low false-positive.
#  * A CVE-shaped token that is not a WELL-FORMED id (CVE-YYYY-NNNN, year >= 1999, >= 4-digit
#    sequence) is a typo or a fabrication -> WARN.
#  * A file HASH is not knowable without computing it, so a 32/40/64-hex hash asserted in the answer
#    that appears in NO successful tool result is very likely invented -> WARN, verify it.
_CVE_LOOSE = re.compile(r"\bCVE-\d{1,4}-\d{1,9}\b", re.I)
_CVE_STRICT = re.compile(r"CVE-(?:1999|20\d{2})-\d{4,7}")
_HASH = re.compile(r"\b(?:[a-fA-F0-9]{64}|[a-fA-F0-9]{40}|[a-fA-F0-9]{32})\b")


def check_cve_format(answer: str) -> list[Finding]:
    out = []
    for m in _CVE_LOOSE.finditer(answer or ""):
        tok = m.group(0)
        if not _CVE_STRICT.fullmatch(tok.upper()):
            out.append(Finding("warn", "cve_format",
                               f"'{tok}' is not a well-formed CVE id (expected CVE-YYYY-NNNN, year "
                               ">= 1999, 4+ digit sequence) - typo or fabrication", tok))
    return out


def check_claim_grounding(answer: str, tool_results: list[dict] | None) -> list[Finding]:
    oks = [r for r in (tool_results or []) if (r.get("result") or {}).get("ok")]
    if not oks:
        return []                       # no successful output to check against
    import json as _json
    corpus = " ".join(_json.dumps(r.get("result")) for r in oks).lower()
    out = []
    for tok in sorted({h for h in _HASH.findall(answer or "")}):
        if tok.lower() not in corpus:
            out.append(Finding("warn", "claim_grounding",
                               f"the answer states a hash '{tok[:12]}...' that is in NO tool "
                               "result - a hash cannot be known without computing it; verify", tok))
    return out


# ---- 7. research discipline: overclaim, unsupported severity, target authorization ----------
# The research-agent rule "never treat a hypothesis as a confirmed vulnerability" (research/findings
# enforces it for STRUCTURED findings) also has to hold for FREE-TEXT answers. These are HEURISTIC
# and WARN-level: they surface a calibration/scope issue, they do not overrule. Every one is
# HEDGE-SUPPRESSED (if the answer already says "likely / possible / would need to verify ...", it is
# being appropriately cautious and nothing fires) and EVIDENCE-AWARE (a real successful tool result
# suppresses the overclaim/severity flags), so they stay quiet on ordinary security Q&A.
_HEDGE = re.compile(
    r"\b(hypothes|likel|possibl|potential|unconfirmed|not\s+confirmed|may\s+be|might\s+be|"
    r"could\s+be|would\s+need|to\s+verify|to\s+confirm|suspect|appears?\s+to|seems?\s+to|"
    r"candidate|unverified|not\s+proven|before\s+confirming|cannot\s+confirm|not\s+yet|"
    r"needs?\s+(?:validation|testing|verifying))\b", re.I)
_CONFIRM_VULN = re.compile(
    r"\b(confirmed|verified|proven|demonstrated)\b[^.\n]{0,45}\b(vulnerab|exploit|rce|inject|"
    r"bypass|traversal|disclosure|ssrf|xxe|idor|overflow)\w*", re.I)
_IS_VULN_ASSERT = re.compile(
    r"\bis\s+(?:definitely|certainly)\s+vulnerable\b|\bis\s+a\s+confirmed\s+(?:vulnerab|rce)\w*", re.I)
_EXPLOIT_SUCCESS = re.compile(
    r"\b(successfully\s+exploited|gained\s+(?:a\s+)?(?:root|shell|access)|achieved\s+rce|"
    r"popped\s+a\s+shell|obtained\s+(?:a\s+)?shell|established\s+persistence|full\s+compromise)\b",
    re.I)
_SEVERITY = re.compile(
    r"\b(?:critical|high)[-\s]?(?:severity|risk)\b|\bseverity[:\s]+(?:critical|high)\b|"
    r"\bcvss[:\s]*(?:score\s*(?:of\s*)?)?(?:9|10|8)(?:\.\d)?\b", re.I)
_ASSESSMENT_CTX = re.compile(
    r"\b(the target|this host|this endpoint|this server|we\s+(?:found|confirmed|identified|"
    r"observed)|affected\s+component|proof[- ]of[- ]concept|reproduc)\b", re.I)
# an active, already-happened action pointed DIRECTLY at a target token (kept tight to avoid noise)
_ACTION_TARGET = re.compile(
    r"\b(scanned|nmapped|exploited|attacked|fuzzed|brute[-\s]?forced|probed|authenticated\s+to|"
    r"connected\s+to)\s+(?:the\s+|host\s+|target\s+)?"
    r"((?:\d{1,3}\.){3}\d{1,3}|https?://[^\s]+|(?:[a-z0-9-]+\.)+[a-z]{2,})", re.I)


def _hedged(answer: str) -> bool:
    return bool(_HEDGE.search(answer or ""))


def _has_ok_tool(tool_results) -> bool:
    return any((r.get("result") or {}).get("ok") for r in (tool_results or []))


def check_overclaim(answer: str, tool_results: list[dict] | None = None) -> list[Finding]:
    """A vulnerability asserted as CONFIRMED/exploited in free text, with no validating tool result
    and no hedge -> WARN to downgrade or validate. Surfaces; does not overrule."""
    a = answer or ""
    if _hedged(a) or _has_ok_tool(tool_results):
        return []
    out = []
    m = _CONFIRM_VULN.search(a) or _IS_VULN_ASSERT.search(a)
    if m:
        out.append(Finding("warn", "overclaim",
                           "states a vulnerability as CONFIRMED/proven, but no validating tool result "
                           "is present and the text does not hedge - downgrade to LIKELY/POSSIBLE or "
                           "run a non-destructive validating test", m.group(0)[:80]))
    e = _EXPLOIT_SUCCESS.search(a)
    if e:
        out.append(Finding("warn", "overclaim",
                           "claims exploitation SUCCESS with no successful tool result to support it "
                           "- treat as unproven until a validating run confirms it", e.group(0)[:80]))
    return out


def check_unsupported_severity(answer: str, tool_results: list[dict] | None = None) -> list[Finding]:
    """A high/critical severity asserted ABOUT A SPECIFIC ASSESSMENT (not a general class fact),
    unhedged and unbacked by a tool result -> WARN. Tied to assessment context so it stays quiet on
    educational answers like 'SQLi is a critical-severity class'."""
    a = answer or ""
    if _hedged(a) or _has_ok_tool(tool_results):
        return []
    sev = _SEVERITY.search(a)
    if sev and _ASSESSMENT_CTX.search(a):
        return [Finding("warn", "severity",
                       "asserts a high/critical severity for this assessment without a validating "
                       "result - unvalidated severity should be labelled ASSERTED and backed by "
                       "evidence before reporting", sev.group(0)[:60])]
    return []


def check_target_authorization(answer: str, authorized_targets: list | None) -> list[Finding]:
    """If the answer claims an active action performed DIRECTLY against a target that is NOT in the
    operator's authorized_targets, surface it (WARN) - either a phantom claim or an out-of-scope
    action. Only runs when authorized_targets is provided; target extraction is kept tight (verb
    immediately followed by an IP/URL/host) to avoid crying wolf on hosts merely mentioned."""
    if authorized_targets is None:
        return []
    from mcp_layer.security import target_authorized
    out, seen = [], set()
    for m in _ACTION_TARGET.finditer(answer or ""):
        verb, tgt = m.group(1), m.group(2).rstrip(".,);")
        if tgt in seen:
            continue
        seen.add(tgt)
        ok, _why = target_authorized(tgt, authorized_targets)
        if not ok:
            out.append(Finding("warn", "target_authorization",
                               f"answer claims an active action ('{verb} {tgt}') against a target "
                               "NOT in authorized_targets - if this ran it was out of scope; verify "
                               "and confirm authorization", f"{verb} {tgt}"))
    return out


# ---- orchestrator -----------------------------------------------------------
def verify(answer: str, hits: list[dict] | None = None,
           tools_ran: list[str] | None = None,
           tool_results: list[dict] | None = None,
           authorized_targets: list | None = None) -> Report:
    """Run every check. `hits` = retrieved RAG chunks; `tools_ran` = tools that executed OK;
    `tool_results` = the actual {tool, result} records, used to catch fabricated tool output;
    `authorized_targets` = the operator's authorization registry (list of dicts) — when supplied,
    an answer claiming an active action against an unauthorized target is flagged."""
    r = Report()
    r.add(*check_math(answer))
    r.add(*check_code(answer))
    r.add(*check_grounding(answer, hits))
    r.add(*check_phantom_actions(answer, tools_ran))
    r.add(*check_tool_grounding(answer, tool_results))
    r.add(*check_cve_format(answer))
    r.add(*check_claim_grounding(answer, tool_results))
    r.add(*check_overclaim(answer, tool_results))
    r.add(*check_unsupported_severity(answer, tool_results))
    r.add(*check_target_authorization(answer, authorized_targets))
    return r


if __name__ == "__main__":
    import sys
    txt = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[1:])
    print(verify(txt).render())
