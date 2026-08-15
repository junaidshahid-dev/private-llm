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
    kind: str           # "math" | "code" | "grounding" | "phantom_action" | "contradiction"
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


# ---- orchestrator -----------------------------------------------------------
def verify(answer: str, hits: list[dict] | None = None,
           tools_ran: list[str] | None = None) -> Report:
    """Run every check. `hits` = retrieved RAG chunks; `tools_ran` = tools actually executed."""
    r = Report()
    r.add(*check_math(answer))
    r.add(*check_code(answer))
    r.add(*check_grounding(answer, hits))
    r.add(*check_phantom_actions(answer, tools_ran))
    return r


if __name__ == "__main__":
    import sys
    txt = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[1:])
    print(verify(txt).render())
