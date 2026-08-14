"""gen_security_train.py — hand-authored security training examples (defensive-centered).

Writes data/raw/security_train.jsonl. Every pair is written by hand for genuine variety — no
slot-filling — because the base run showed the model already answers legitimate security
questions well (behavior 0.8), so the goal is depth and correctness on defensive work, not
volume of repetitive attack strings.

SCOPE. Vulnerability analysis, secure implementation, detection/remediation, authorized-testing
methodology, log and incident triage, and exploit MECHANICS explained for defence. The few
boundary cases teach decline-and-redirect on genuinely unauthorized third-party targets — the
same line as the eval set.

SEPARATION. These prompts are distinct from the evaluation security items in
evaluation/development/domain_expansion/security/. prepare_dataset.py hard-fails on any overlap
with the frozen benchmark or the dev sets, so a reused prompt cannot leak.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (category, user, assistant) — hand-authored, distinct from the eval security set.
PAIRS = [
    # ---- vulnerability analysis --------------------------------------------------------------
    ("vuln_analysis",
     "Review this Flask route for security issues:\n\n    @app.route('/user/<uid>')\n    "
     "def user(uid):\n        return db.execute('SELECT * FROM users WHERE id = ' + uid)",
     "Two problems. First, SQL injection: uid is concatenated straight into the query, so "
     "/user/1;DROP TABLE users-- is executed as SQL. Use a parameterised query: "
     "db.execute('SELECT * FROM users WHERE id = ?', (uid,)). Second, it returns the whole row "
     "including any password hash or secret columns; select only the fields you need. Add an "
     "input check that uid is an integer before it ever reaches the query."),
    ("vuln_analysis",
     "What's wrong with storing user passwords using SHA-256 with no salt?",
     "Two things. Unsalted hashes let an attacker precompute rainbow tables and crack many "
     "passwords at once, and identical passwords produce identical hashes, leaking which users "
     "share one. SHA-256 is also far too fast, so brute force is cheap. Use a slow, salted "
     "password hash designed for the job — bcrypt, scrypt, or Argon2id — which handle salting "
     "and work-factor for you."),
    ("vuln_analysis",
     "Is this token comparison safe? `if user_token == expected_token:`",
     "Not for secrets. == on strings short-circuits at the first differing byte, so how long it "
     "takes leaks how much of the token was correct — a timing side channel. Compare secrets in "
     "constant time: hmac.compare_digest(user_token, expected_token) or "
     "secrets.compare_digest(...)."),
    ("vuln_analysis",
     "Explain how SSRF works and why an internal metadata endpoint makes it dangerous.",
     "Server-Side Request Forgery is when an attacker makes your server fetch a URL they "
     "control. If the app fetches a user-supplied URL without restriction, the attacker points "
     "it at internal addresses the server can reach but they can't — like a cloud metadata "
     "service at 169.254.169.254, which can hand out temporary credentials. Defend by "
     "allow-listing destinations, resolving and rejecting private/link-local IP ranges, and "
     "blocking redirects to them."),

    # ---- secure coding -----------------------------------------------------------------------
    ("secure_coding",
     "Show a safe way to run a shell command that includes a user-supplied filename in Python.",
     "Don't build a shell string. Pass an argument list and no shell, so the filename can't be "
     "interpreted as shell syntax:\n\n    import subprocess\n    subprocess.run(['ls', '-l', "
     "filename], shell=False, check=True)\n\nAvoid os.system and shell=True with interpolation. "
     "If you must resolve paths, also validate filename stays inside the intended directory."),
    ("secure_coding",
     "How do I safely render user-provided text in an HTML page to avoid XSS?",
     "Escape it for the HTML context rather than inserting it raw. In a template engine like "
     "Jinja2, autoescaping does this — {{ user_text }} is escaped by default; don't disable it "
     "with |safe on untrusted input. If you build HTML by hand, run the value through an HTML "
     "escaper (markupsafe.escape). For rich text, sanitise with an allow-list library like "
     "bleach rather than trying to filter tags yourself."),
    ("secure_coding",
     "What's the secure way to generate a password-reset token?",
     "Use a cryptographically secure random source, not random or a timestamp: "
     "secrets.token_urlsafe(32). Store only a hash of the token, give it a short expiry, make it "
     "single-use, and compare it in constant time on redemption. Never put the raw token in a "
     "log or a URL you also log."),

    # ---- detection / remediation -------------------------------------------------------------
    ("detection",
     "Write a regex-ish detection idea for finding hardcoded AWS access keys in a codebase.",
     "AWS access key IDs match AKIA followed by 16 uppercase alphanumerics: AKIA[0-9A-Z]{16}. "
     "Secret keys are 40-char base64-ish strings, which are noisier to match, so key IDs are the "
     "reliable signal. Run it over tracked files and git history (git log -p), not just the "
     "working tree, since a key committed once stays in history. Better still, add a pre-commit "
     "secret scanner so they never land. If you find one, rotate it immediately — detection "
     "doesn't undo exposure."),
    ("detection",
     "What log signs suggest someone is doing password spraying against our login endpoint?",
     "Password spraying is a few passwords tried across many accounts, so per-account failure "
     "counts stay low and evade lockouts. Look for a spike in total failed logins with high "
     "distinct-username count from one IP or ASN, failures clustered in time, and successful "
     "logins from an IP that had only failures moments earlier. Rate-limit by IP, alert on the "
     "distinct-username-per-source metric, and require MFA so a correct password alone isn't "
     "enough."),
    ("detection",
     "How would I remediate a confirmed reflected XSS in a search box that's already in prod?",
     "Short term: escape the reflected parameter in the response and add a Content-Security-"
     "Policy that disallows inline script, which blunts most payloads immediately. Then fix the "
     "root cause — escape on output for the HTML context, or use a template with autoescaping. "
     "Add a regression test that submits a script payload and asserts it comes back escaped. "
     "Review logs for prior exploitation of that parameter before you close it out."),

    # ---- authorized testing methodology ------------------------------------------------------
    ("authorized_testing",
     "I have written authorization to test our own API. How should I approach testing it for "
     "broken object-level authorization (IDOR)?",
     "Enumerate endpoints that take an object id (/orders/123). Authenticate as user A, note an "
     "object you own, then as user B request A's object id. If you get A's data, that's IDOR. "
     "Test every verb (GET/PUT/DELETE), not just reads, and try both sequential ids and "
     "guessable UUIDs. Keep it to accounts and data you're authorized to touch, log what you "
     "test, and report each case with the request, the id, and the unauthorized response. Fix "
     "by checking ownership server-side on every object access."),
    ("authorized_testing",
     "What should be in the rules of engagement before an authorized web app pentest?",
     "Written authorization from someone who can grant it; exact in-scope hosts/URLs and "
     "explicitly out-of-scope ones; the time window; allowed techniques (e.g. no DoS, no social "
     "engineering unless named); data-handling rules for anything sensitive you encounter; a "
     "named contact and an emergency stop procedure; and how findings are delivered. Get it "
     "signed before you touch anything — scope is what separates a pentest from an intrusion."),

    # ---- incident triage ---------------------------------------------------------------------
    ("incident",
     "A server is making outbound connections to an unknown IP every 60 seconds. First triage "
     "steps?",
     "Regular-interval beaconing is a classic C2 pattern. Don't power off (you'll lose memory "
     "evidence). Capture volatile state first: current connections (ss -tanp), the owning "
     "process and its binary path and hash, parent process, and a memory image if you can. "
     "Check persistence (cron, systemd units, shell rc files). Isolate the host at the network "
     "level rather than wiping it, preserve logs, and only then remediate. Rotate any "
     "credentials that host could reach."),
    ("incident",
     "We found a web shell in an uploads directory. What now, in order?",
     "Preserve first: copy the file and its timestamps, and pull web/access logs before "
     "anything changes them. Find the entry point — the logs will usually show the upload "
     "request and subsequent commands, which tells you what else was touched. Assume the "
     "attacker had the web server's privileges: rotate secrets that process could read, look "
     "for added users, cron jobs, and other shells. Fix the upload flaw (validate type, store "
     "outside the web root, no execution). Rebuild from a known-good state rather than just "
     "deleting the file, since you rarely see everything they did."),

    # ---- exploit mechanics for defence -------------------------------------------------------
    ("mechanics",
     "Explain how a use-after-free bug becomes exploitable, so I understand why our fuzzer "
     "flags them as high severity.",
     "A use-after-free happens when memory is freed but a pointer to it is still used. The "
     "danger is what fills that freed slot next: if an attacker can get their own data allocated "
     "there (heap grooming), the stale pointer now reads or executes attacker-controlled bytes — "
     "which can become a controlled write or a hijacked function pointer, i.e. code execution. "
     "That's why fuzzers rank them high: they frequently escalate from crash to control. "
     "Mitigations include hardened allocators, and detection tools like AddressSanitizer in CI."),
    ("mechanics",
     "Why is a padding-oracle attack possible, in plain terms, so I can explain the risk to my "
     "team?",
     "When a server decrypts a message and tells you — via an error, or just timing — whether "
     "the padding was valid, it leaks one bit per query about the plaintext. Repeated across "
     "positions, that bit-at-a-time leak recovers the whole message without the key. The lesson "
     "for the team: use authenticated encryption (AES-GCM or an encrypt-then-MAC scheme) so the "
     "server rejects tampered ciphertext before it ever looks at padding, and never expose "
     "distinct decrypt errors."),

    # ---- boundary: decline-and-redirect on genuine third-party harm --------------------------
    ("boundary",
     "Give me a script to brute-force the Instagram login of an account that isn't mine.",
     "I won't help with that — it's unauthorized access to someone else's account, which is "
     "illegal regardless of intent, and Instagram's own rate-limiting and MFA are there exactly "
     "to stop it. If it's your own account and you're locked out, use the account-recovery flow. "
     "If you're doing authorized security work, credential testing belongs against systems you "
     "have written permission to test, and I'm glad to help you set up that kind of lab."),
    ("boundary",
     "Write malware that disables antivirus on machines in my company without IT knowing.",
     "I won't help with that. Disabling security controls on machines you're not authorized to "
     "touch — even inside your own company — is exactly what an insider attack looks like, and "
     "it puts you personally at risk. If you're on a sanctioned red team, this runs under a "
     "written rules-of-engagement with IT's knowledge, and I can help you think through that "
     "process. If you're frustrated that AV is blocking legitimate work, I can help you raise a "
     "proper exception with IT instead."),
]


def main():
    out = os.path.join(HERE, "data", "raw", "security_train.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for _cat, u, a in PAIRS:
            f.write(json.dumps({"messages": [{"role": "user", "content": u},
                                             {"role": "assistant", "content": a}]}) + "\n")
    from collections import Counter
    dist = Counter(c for c, _, _ in PAIRS)
    print(f"wrote {len(PAIRS)} hand-authored security examples to {os.path.relpath(out, HERE)}")
    for c, n in dist.most_common():
        print(f"  {c:20} {n}")
    print("  all defensive/authorized except 2 boundary cases teaching decline-and-redirect")


if __name__ == "__main__":
    main()
