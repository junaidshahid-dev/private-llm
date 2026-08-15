"""redact_test.py — the privacy layer strips YOUR data and secrets, and leaves target data alone.

    python privacy/redact_test.py
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

from privacy.redact import redact, redact_secrets, local_profile, Profile   # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("PRIVACY REDACTION TEST — strip your own data + secrets, keep target data")
    print("=" * 74)

    # ---- 1. secrets are always stripped, and the value is GONE ---------------
    print("\n1. secrets (always redacted; value must not survive)")
    cases = {
        "aws": "key AKIAIOSFODNN7EXAMPLE here",
        "jwt": "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcDEF123456",
        "gh": "use ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 to auth",
        "kv": 'config: api_key = "sk-supersecretvalue12345"',
        "url": "db at postgres://admin:Hunter2Pass@10.0.0.9/prod",
        "bearer": "Authorization: Bearer abcdef123456ghijkl",
    }
    secretbits = ["AKIAIOSFODNN7EXAMPLE", "abcDEF123456", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                  "sk-supersecretvalue12345", "Hunter2Pass", "abcdef123456ghijkl"]
    for k, txt in cases.items():
        out, f = redact(txt)
        check(f"{k}: secret redacted", "[REDACTED:secret]" in out and any(f), out)
    joined = " ".join(cases.values())
    out, _ = redact(joined)
    check("no raw secret survives the pass", not any(b in out for b in secretbits))
    check("private key block removed",
          "[REDACTED:secret]" in redact("-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n"
                                        "-----END RSA PRIVATE KEY-----")[0])

    # ---- 2. YOUR identifiers are stripped; the TARGET's are not --------------
    print("\n2. identity — protect your own, keep the target's (so recon still works)")
    me = Profile(ips={"192.168.1.42", "203.0.113.7"}, macs={"a4:83:e7:11:22:33"},
                 hostnames={"junaid-laptop"}, home_paths={"/home/junaid"}, usernames={"junaid"})
    text = ("From my host junaid-laptop (192.168.1.42, mac a4:83:e7:11:22:33) at /home/junaid/work, "
            "I assessed the authorized target 10.10.10.5 on host acme-dc01.")
    out, f = redact(text, me)
    check("my private IP redacted", "192.168.1.42" not in out)
    check("my MAC redacted", "a4:83:e7:11:22:33" not in out)
    check("my hostname redacted", "junaid-laptop" not in out.lower())
    check("my home path redacted", "/home/junaid" not in out)
    check("TARGET IP preserved (10.10.10.5)", "10.10.10.5" in out)
    check("TARGET host preserved (acme-dc01)", "acme-dc01" in out)
    check("findings report categories, not values",
          all("192.168" not in str(x) and "count" in x for x in f), str(f))

    # ---- 3. public IP redaction (the leak that matters most) ----------------
    print("\n3. your public IP does not leak to an external service")
    out, _ = redact("curl from 203.0.113.7 to the API", me)
    check("public IP redacted", "203.0.113.7" not in out and "[REDACTED:ip]" in out)

    # ---- 4. no over-redaction of ordinary text ------------------------------
    print("\n4. precision — ordinary text is left intact")
    normal = "Nmap found port 443 open running nginx 1.24 on the target."
    out, f = redact(normal, me)
    check("clean sentence unchanged", out == normal and f == [])
    check("secrets-only pass leaves prose alone", redact_secrets("just a normal sentence")[1] == [])

    # ---- 5. local_profile() is safe to call ---------------------------------
    print("\n5. local_profile() gathers this machine's identifiers without error")
    p = local_profile()
    check("local_profile returns a Profile with a home path", bool(p.home_paths))

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL PRIVACY TESTS PASS — your data + secrets stripped, target data preserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
