"""gen_security_train.py — hand-authored security training examples (defensive-centered).

Writes data/raw/security_train.jsonl. Every pair is written by hand for genuine variety — no
slot-filling — because the base run showed the model already answers legitimate security
questions well (behavior 0.8), so the goal is depth and correctness on defensive work, not
volume of repetitive attack strings.

SCOPE. Pure capability: vulnerability analysis, secure implementation, detection/remediation,
authorized-testing methodology, log and incident triage, and exploit MECHANICS. Security topic
knowledge (brute-force, malware internals, AV/EDR, exploitation, privesc, RE, offensive tooling,
vuln research) is preserved, never degraded.

NO BEHAVIOURAL POLICY IN THE WEIGHTS. This dataset teaches capability and factuality, not
topic-based refusals. Any decline policy the operator wants lives in a separate, configurable
inference-time layer (a system-prompt policy), so it stays modular and does not bias the model
into keyword-triggered refusals. What this file will NOT contain, by author's choice, is data
that actively walks through executing an attack on a specific third party (brute-forcing a
stranger's account, deploying malware to a system you don't control) — explaining a technique is
capability; scripting an attack on a named victim is not, and that stays out of the training
data regardless of the policy layer.

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

    # NOTE: behavioural policy (what to decline) is deliberately NOT baked into this capability
    # dataset. Topic knowledge — brute-force, malware internals, AV/EDR, exploitation, privesc —
    # stays intact; any decline policy is a separate, operator-configurable layer at inference
    # (see the system-prompt policy scaffold), so it can be changed without retraining and does
    # not bias the model into keyword-based refusals. The examples below are all capability.

    # ---- more secure coding ------------------------------------------------------------------
    ("secure_coding",
     "How should I store a third-party API key that my Python service needs at runtime?",
     "Keep it out of the code and out of git. Read it from an environment variable or a secrets "
     "manager (AWS Secrets Manager, Vault, or the platform's secret store) at startup. Add the "
     ".env file to .gitignore, and if a key was ever committed, rotate it — git history keeps it "
     "even after you delete the line. Never log the key, and scope it to the least privilege the "
     "service actually needs."),
    ("secure_coding",
     "Is it safe to build a MongoDB query from a dict I got from request JSON?",
     "Not directly — NoSQL injection is real. If a field's value is itself an object like "
     "{'$ne': None} or {'$gt': ''}, it becomes a query operator and can bypass your filter. "
     "Coerce each field to the expected scalar type before it reaches the query (str(username), "
     "not the raw value), and validate the shape with a schema. Don't pass user-controlled dicts "
     "straight into find()."),
    ("secure_coding",
     "What's the right way to set a session cookie so it resists theft?",
     "Set HttpOnly so JavaScript can't read it (blunts XSS token theft), Secure so it only goes "
     "over HTTPS, and SameSite=Lax or Strict to limit cross-site sending (blunts CSRF). Give it "
     "a sensible expiry, scope Path/Domain tightly, and rotate the session id on login to "
     "prevent fixation."),
    ("secure_coding",
     "How do I validate a file upload so it can't be used to attack my server?",
     "Don't trust the client's filename or content-type. Generate your own storage name, store "
     "outside the web root so the file can't be executed as a script, and enforce a size limit. "
     "Check the actual content against an allow-list of types (magic bytes, not the extension). "
     "For images, re-encode them through a library, which strips embedded payloads. Serve them "
     "back with a fixed Content-Type and Content-Disposition."),

    # ---- more authorized testing / analysis --------------------------------------------------
    ("authorized_testing",
     "During an authorized test, what's a safe way to check whether a form is vulnerable to "
     "reflected XSS without doing damage?",
     "Use a harmless, unique marker rather than a real payload. Submit something like "
     "zzxss<i>123</i> and look at whether it comes back rendered as a tag or escaped as text in "
     "the response HTML. A non-executing probe like that proves reflection and escaping "
     "behaviour without running script. Record the request, the parameter, and the raw response "
     "for the report, and only test forms in your authorized scope."),
    ("authorized_testing",
     "How do I responsibly disclose a vulnerability I found in an open-source project?",
     "Look for a SECURITY.md or a security contact / GitHub private advisory; use that rather "
     "than a public issue. Send a clear report: affected version, steps to reproduce, impact, "
     "and a suggested fix if you have one. Give them reasonable time to respond and patch "
     "(commonly 90 days) before any public write-up, and coordinate the disclosure date with "
     "them. Don't test against their production systems to confirm it."),
    ("authorized_testing",
     "What does a good vulnerability report to a client contain?",
     "A clear title and severity (with a CVSS vector if you use one), the affected asset, "
     "concrete reproduction steps a developer can follow, the concrete impact ('an "
     "unauthenticated user can read any other user's orders'), and a specific remediation. "
     "Include just enough evidence to prove it — the request and response — and avoid dumping "
     "real sensitive data you accessed. Rank findings so they can fix the worst first."),

    # ---- more detection / incident -----------------------------------------------------------
    ("detection",
     "What in our web logs would indicate someone is fuzzing for hidden endpoints?",
     "A burst of 404s from one source hitting many distinct paths, often from a wordlist "
     "(/admin, /.git/config, /backup.zip, /.env), sometimes with a scanner User-Agent. Watch "
     "the distinct-404-paths-per-source metric and sudden 403/401 clusters. Alert on it, make "
     "sure sensitive files (.env, .git) aren't served, and return uniform responses so you don't "
     "confirm which paths exist."),
    ("incident",
     "A developer accidentally committed an AWS key to a public repo. What's the response order?",
     "Treat it as already compromised — bots scrape public commits within minutes. First, "
     "deactivate/rotate the key in IAM immediately; that stops the bleeding. Then review "
     "CloudTrail for any use of that key you didn't expect. Remove it from the code AND purge it "
     "from git history (or, more reliably, rotate and consider the old key permanently burned). "
     "Add a pre-commit secret scanner so it can't recur."),
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
    print("  all capability — no baked-in refusal policy; security knowledge preserved")


if __name__ == "__main__":
    main()
