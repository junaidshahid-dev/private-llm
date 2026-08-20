"""static_test.py — the source analyzers detect real issues and DON'T over-flag safe code.

    python analysis/static_test.py
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from analysis.static import (scan_secrets, scan_dangerous_apis, taint_analysis,     # noqa: E402
                             analyze_python)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def has_class(findings, vclass):
    return any(f.get("vuln_class") == vclass for f in findings)


def main() -> int:
    print("=" * 74)
    print("STATIC ANALYSIS — secrets, dangerous APIs, and input->sink taint")
    print("=" * 74)

    print("\n1. secret detection (with placeholder filtering)")
    s = scan_secrets('AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\npassword = "changeme"\napi_key = "s3cr3tLiveValue123"')
    kinds = {x["kind"] for x in s}
    check("finds an AWS access key", "aws_access_key" in kinds, str(kinds))
    check("finds a real assigned secret", "assigned_secret" in kinds)
    check("placeholder 'changeme' is NOT reported", not any("changeme" in x["match"] for x in s))
    check("the reported secret value is redacted", all("s3cr3tLiveValue123" != x["match"] for x in s))
    check("a private key block is detected",
          any(x["kind"] == "private_key" for x in scan_secrets("-----BEGIN RSA PRIVATE KEY-----")))

    print("\n2. dangerous API detection")
    d = scan_dangerous_apis("import os\nos.system(x)\neval(y)\nsubprocess.run(c, shell=True)\n"
                            "# eval(this_is_a_comment)\n")
    names = {x["name"] for x in d}
    check("flags os.system", "os.system" in names)
    check("flags eval", "eval" in names)
    check("flags subprocess shell=True", "subprocess shell=True" in names)
    check("does NOT flag eval inside a comment", not any(x["line"] == 5 for x in d), str(d))
    js = scan_dangerous_apis("el.innerHTML = userInput;\neval(z);", "javascript")
    check("flags innerHTML (js)", any(x["name"] == "innerHTML" for x in js))

    print("\n3. taint: attacker input reaching a sink")
    t = taint_analysis("cmd = request.args.get('c')\nos.system(cmd)")
    check("command injection via request -> os.system", has_class(t, "command_injection"), str(t))
    check("reports source and sink lines", t and t[0]["source_line"] == 1 and t[0]["sink_line"] == 2)
    t2 = taint_analysis("os.system(request.args['c'])")
    check("direct source-in-sink is caught", has_class(t2, "command_injection"))
    t3 = taint_analysis("q = \"SELECT * FROM u WHERE n='\" + request.args.get('n') + \"'\"\n"
                        "cursor.execute(q)")
    check("sql injection via string-built query -> .execute", has_class(t3, "sql_injection"))
    t4 = taint_analysis("import os\ncmd = request.args.get('c')\nos.system(shlex.quote(cmd))")
    # shlex.quote sanitizes the value passed to the sink
    check("sanitized value (shlex.quote at the sink) is NOT flagged", not t4, str(t4))
    t5 = taint_analysis("os.system('ls -la')")
    check("constant argument is NOT flagged", not t5)

    print("\n4. taint does NOT flag a PARAMETERIZED query (the key false-positive to avoid)")
    param = taint_analysis("cursor.execute('SELECT * FROM u WHERE n=%s', (request.args.get('n'),))")
    check("parameterized query with tainted PARAMS is safe (query arg is constant)", not param, str(param))

    print("\n5. malformed code does not crash the analyzer")
    check("syntax error -> empty taint result, no exception", taint_analysis("def (:\n  x=") == [])

    print("\n6. analyze_python composes HYPOTHESES (never confirmed) that rank")
    hyps = analyze_python("cmd = request.args.get('c')\nos.system(cmd)\n"
                          'API_KEY = "AKIAIOSFODNN7EXAMPLE"', filename="app.py")
    check("produces hypotheses", len(hyps) >= 2, str(len(hyps)))
    check("a static finding is never CONFIRMED (needs a validating test)",
          all(h.status != "CONFIRMED" for h in hyps), str([h.status for h in hyps]))
    check("each carries a next_test and a code-evidence source",
          all(h.next_test and h.evidence and h.evidence[0].kind == "code" for h in hyps))
    check("affected component points at file:line", any(":" in h.affected_component for h in hyps))

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL STATIC-ANALYSIS TESTS PASS — finds real sinks/secrets, spares safe code (incl.")
    print("parameterized queries), and emits hypotheses that require validation, never 'confirmed'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
