"""static.py — read-only source analysis: secrets, dangerous APIs, and input->sink taint.

Passive by nature: it inspects code, it never touches a target, so it sits behind the same read-only
filesystem permission as fs_read. Three analyzers, all pure:

  scan_secrets        hardcoded credentials (keys, private keys, tokens, secret= assignments), with
                      placeholder filtering so 'password = "changeme"' is not reported.
  scan_dangerous_apis dangerous sinks (eval/exec/os.system/subprocess shell=True/pickle/yaml.load/
                      verify=False/innerHTML/...), each with WHY and a vulnerability class.
  taint_analysis      Python AST intraprocedural data-flow: does attacker-controlled input (request.*,
                      input(), os.environ, sys.argv) reach a dangerous sink UNSANITIZED? It checks the
                      sink's dangerous ARGUMENT only, so a parameterized query (tainted value in the
                      params, constant in the query) is correctly NOT flagged.

analyze_python() composes the results into research.findings HYPOTHESES — evidence kind="code", so a
static hit is POSSIBLE/LIKELY and NEVER CONFIRMED without a validating test. Honest limits: taint is
intraprocedural, flow-union (no cross-function/alias tracking), so it is a lead generator, not proof.
"""
from __future__ import annotations

import ast
import re

from research.findings import Evidence, Hypothesis

# ---- 1. secrets ------------------------------------------------------------------------------
_SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}\b")),
    ("assigned_secret", re.compile(
        r"(?i)\b(api[_-]?key|secret(?:[_-]?key)?|access[_-]?key|client[_-]?secret|auth[_-]?token|"
        r"token|passwd|password)\b\s*[:=]\s*['\"]([^'\"]{6,})['\"]")),
]
_PLACEHOLDER = re.compile(
    r"(?i)^(?:changeme|change_me|your[_-]?\w+|example|test|dummy|placeholder|xxx+|todo|none|null|"
    r"<[^>]+>|\{\{?[^}]+\}?\}|\$\{[^}]+\}|\.{3,}|sk-\.\.\.|redacted)$")


def _redact(s: str) -> str:
    return (s[:4] + "…" + s[-2:]) if len(s) > 8 else "…"


def scan_secrets(code: str) -> list[dict]:
    out, seen = [], set()
    for i, line in enumerate(code.splitlines(), 1):
        for name, rx in _SECRET_PATTERNS:
            m = rx.search(line)
            if not m:
                continue
            val = m.group(2) if (name == "assigned_secret" and m.lastindex and m.lastindex >= 2) else m.group(0)
            if name == "assigned_secret" and (_PLACEHOLDER.match(val.strip()) or len(set(val)) <= 2):
                continue                              # obvious placeholder, not a real secret
            key = (name, i)
            if key in seen:
                continue
            seen.add(key)
            out.append({"line": i, "kind": name, "severity": "HIGH",
                        "match": _redact(val), "why": "hardcoded secret in source (and in VCS history)"})
    return out


# ---- 2. dangerous APIs -----------------------------------------------------------------------
# (regex, name, severity, why, vuln_class). Full-line comments are stripped before matching.
_DANGER = {
    "python": [
        (r"\beval\s*\(", "eval", "CRITICAL", "evaluates arbitrary code from its argument", "code_execution"),
        (r"\bexec\s*\(", "exec", "CRITICAL", "executes arbitrary code from its argument", "code_execution"),
        (r"\bos\.system\s*\(", "os.system", "HIGH", "runs a shell command", "command_injection"),
        (r"\bos\.popen\s*\(", "os.popen", "HIGH", "runs a shell command", "command_injection"),
        (r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True", "subprocess shell=True", "HIGH",
         "spawns a shell — injectable if any part is user-controlled", "command_injection"),
        (r"\bpickle\.loads?\s*\(", "pickle.load", "HIGH", "deserializing untrusted data is RCE",
         "insecure_deserialization"),
        (r"\byaml\.load\s*\((?![^)]*Safe)", "yaml.load", "HIGH",
         "yaml.load without SafeLoader can execute code", "insecure_deserialization"),
        (r"\bmarshal\.loads?\s*\(", "marshal.load", "HIGH", "deserializes untrusted bytecode",
         "insecure_deserialization"),
        (r"verify\s*=\s*False", "tls verify=False", "HIGH", "disables TLS certificate verification",
         "crypto_misuse"),
        (r"debug\s*=\s*True", "debug=True", "MEDIUM", "debug mode leaks internals / can allow RCE",
         "misconfiguration"),
        (r"\bhashlib\.(?:md5|sha1)\s*\(", "weak hash", "MEDIUM",
         "MD5/SHA1 are weak (unsuitable for passwords/integrity)", "crypto_misuse"),
    ],
    "javascript": [
        (r"\beval\s*\(", "eval", "CRITICAL", "evaluates arbitrary code", "code_execution"),
        (r"new\s+Function\s*\(", "new Function", "HIGH", "constructs code from a string", "code_execution"),
        (r"child_process\.(?:exec|execSync)\s*\(", "child_process.exec", "HIGH",
         "runs a shell command", "command_injection"),
        (r"\.innerHTML\s*=", "innerHTML", "HIGH", "assigning HTML can inject script", "xss"),
        (r"dangerouslySetInnerHTML", "dangerouslySetInnerHTML", "HIGH", "renders raw HTML", "xss"),
        (r"document\.write\s*\(", "document.write", "MEDIUM", "writes raw HTML to the page", "xss"),
    ],
}


def _strip_comment(line: str, lang: str) -> str:
    marker = "//" if lang == "javascript" else "#"
    stripped = line.lstrip()
    return "" if stripped.startswith(marker) else line


def scan_dangerous_apis(code: str, lang: str = "python") -> list[dict]:
    table = _DANGER.get(lang, [])
    out = []
    for i, raw in enumerate(code.splitlines(), 1):
        line = _strip_comment(raw, lang)
        if not line.strip():
            continue
        for pat, name, sev, why, vclass in table:
            if re.search(pat, line):
                out.append({"line": i, "name": name, "severity": sev, "why": why,
                            "vuln_class": vclass, "snippet": raw.strip()[:120]})
    return out


# ---- 3. taint analysis (Python AST, intraprocedural) -----------------------------------------
_SOURCE_ATTRS = ("request.args", "request.form", "request.values", "request.json", "request.data",
                 "request.get_json", "request.cookies", "request.headers", "request.GET",
                 "request.POST", "sys.argv", "os.environ", "flask.request.args")
_SOURCE_CALLS = ("input", "os.getenv", "os.environ.get")
_SANITIZERS = ("int", "float", "bool", "shlex.quote", "html.escape", "escape", "bleach.clean",
               "re.escape", "quote", "secure_filename", "urllib.parse.quote")
# dotted-func / suffix -> vuln class. Only the sink's dangerous ARG (args[0]) is checked.
_SINK_EXACT = {"os.system": "command_injection", "os.popen": "command_injection",
               "eval": "code_execution", "exec": "code_execution", "open": "path_traversal",
               "os.remove": "path_traversal"}
_SINK_SUFFIX = {"execute": "sql_injection", "executemany": "sql_injection",
                "render_template_string": "ssti"}


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _is_source(node) -> bool:
    if isinstance(node, ast.Call):
        fn = _dotted(node.func)
        if fn in _SOURCE_CALLS:
            return True
        if any(fn == s or fn.startswith(s + ".") for s in _SOURCE_ATTRS):
            return True
        return False
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        d = _dotted(node.value if isinstance(node, ast.Subscript) else node)
        return any(d == s or d.startswith(s + ".") for s in _SOURCE_ATTRS)
    return False


def _tainted(node, tainted: set) -> bool:
    if node is None:
        return False
    if _is_source(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Call):
        if _dotted(node.func) in _SANITIZERS or _dotted(node.func).split(".")[-1] in _SANITIZERS:
            return False                               # sanitized -> clean
        return any(_tainted(a, tainted) for a in node.args) \
            or any(_tainted(k.value, tainted) for k in node.keywords)
    if isinstance(node, ast.Attribute):
        return _tainted(node.value, tainted)
    if isinstance(node, ast.Subscript):
        return _tainted(node.value, tainted)
    if isinstance(node, ast.BinOp):
        return _tainted(node.left, tainted) or _tainted(node.right, tainted)
    if isinstance(node, ast.BoolOp):
        return any(_tainted(v, tainted) for v in node.values)
    if isinstance(node, ast.JoinedStr):
        return any(_tainted(v, tainted) for v in node.values)
    if isinstance(node, ast.FormattedValue):
        return _tainted(node.value, tainted)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_tainted(e, tainted) for e in node.elts)
    if isinstance(node, ast.Starred):
        return _tainted(node.value, tainted)
    return False


def _sink_class(call) -> str | None:
    fn = _dotted(call.func)
    if fn in _SINK_EXACT:
        return _SINK_EXACT[fn]
    if fn.startswith("subprocess.") and fn.split(".")[-1] in (
            "call", "run", "Popen", "check_output", "check_call"):
        return "command_injection"
    return _SINK_SUFFIX.get(fn.split(".")[-1])


def _scope_stmts(body):
    """Statements of a scope in source order, descending into control-flow but NOT nested defs."""
    out = []
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        out.append(stmt)
        for field in ("body", "orelse", "finalbody"):
            out.extend(_scope_stmts(getattr(stmt, field, []) or []))
        for h in getattr(stmt, "handlers", []) or []:
            out.extend(_scope_stmts(h.body))
    return out


def taint_analysis(code: str) -> list[dict]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    scopes = [tree.body] + [n.body for n in ast.walk(tree)
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    findings, seen = [], set()
    for body in scopes:
        tainted: set = set()
        origin: dict = {}
        for stmt in _scope_stmts(body):
            # first, any sink CALL in this statement is checked against the current taint set
            for call in [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]:
                vclass = _sink_class(call)
                if vclass and call.args and _tainted(call.args[0], tainted):
                    var = next((a.id for a in ast.walk(call.args[0]) if isinstance(a, ast.Name)
                                and a.id in tainted), None)
                    key = (call.lineno, _dotted(call.func))
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append({
                        "sink": _dotted(call.func), "vuln_class": vclass, "sink_line": call.lineno,
                        "source_line": origin.get(var, call.lineno), "tainted_var": var,
                        "why": f"attacker-controlled input reaches {_dotted(call.func)} unsanitised"})
            # then update taint from assignments in this statement
            if isinstance(stmt, ast.Assign):
                t = _tainted(stmt.value, tainted)
                for tgt in stmt.targets:
                    for nm in [n for n in ast.walk(tgt) if isinstance(n, ast.Name)]:
                        if t:
                            tainted.add(nm.id)
                            origin.setdefault(nm.id, stmt.lineno)
                        else:
                            tainted.discard(nm.id)
            elif isinstance(stmt, ast.AugAssign) and _tainted(stmt.value, tainted):
                for nm in [n for n in ast.walk(stmt.target) if isinstance(n, ast.Name)]:
                    tainted.add(nm.id)
                    origin.setdefault(nm.id, stmt.lineno)
    return findings


# ---- orchestrator: static findings as research HYPOTHESES ------------------------------------
_SEV_IMPACT = {"CRITICAL": 0.9, "HIGH": 0.75, "MEDIUM": 0.5, "LOW": 0.3}


def analyze_python(code: str, filename: str = "source") -> list[Hypothesis]:
    """Compose the analyzers into ranked HYPOTHESES (never confirmed — each needs a validating test)."""
    hyps: list[Hypothesis] = []
    for t in taint_analysis(code):
        loc = f"{filename}:{t['sink_line']}"
        hyps.append(Hypothesis(
            title=f"{t['vuln_class']} — input may reach {t['sink']}",
            vuln_class=t["vuln_class"], affected_component=loc,
            observation=f"taint: source line {t['source_line']} -> sink {t['sink']} at line {t['sink_line']}",
            why_it_matters=t["why"], impact=0.8, exploitability=0.6, cost_to_verify=0.3,
            next_test="confirm attacker-controlled input actually reaches this sink at runtime, "
                      "unsanitised (a non-destructive probe), before claiming exploitable",
            expected_result="the input is reflected/executed at the sink",
            alternative_explanation="the input may be validated/sanitised elsewhere, or the path "
                                    "may be unreachable — intraprocedural taint cannot see that",
            evidence=[Evidence(loc, "code", t["why"], 0.7)]))
    for d in scan_dangerous_apis(code, "python"):
        loc = f"{filename}:{d['line']}"
        hyps.append(Hypothesis(
            title=f"dangerous API: {d['name']}", vuln_class=d["vuln_class"], affected_component=loc,
            observation=d["snippet"], why_it_matters=d["why"],
            impact=_SEV_IMPACT.get(d["severity"], 0.5), exploitability=0.5, cost_to_verify=0.3,
            severity=d["severity"],
            next_test="determine whether any argument is attacker-controlled and unsanitised",
            alternative_explanation="the argument may be a constant / not attacker-controlled",
            evidence=[Evidence(loc, "code", d["snippet"], 0.6)]))
    for s in scan_secrets(code):
        loc = f"{filename}:{s['line']}"
        hyps.append(Hypothesis(
            title=f"hardcoded secret ({s['kind']})", vuln_class="hardcoded_secret",
            affected_component=loc, observation=f"{s['kind']} = {s['match']}", why_it_matters=s["why"],
            impact=0.7, exploitability=0.6, cost_to_verify=0.1, severity="HIGH",
            next_test="confirm the credential is live/valid and rotate it if so",
            alternative_explanation="may be a test/placeholder credential",
            evidence=[Evidence(loc, "code", s["kind"], 0.8)]))
    return hyps
