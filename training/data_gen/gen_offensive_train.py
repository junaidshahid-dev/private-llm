"""gen_offensive_train.py — hand-authored OFFENSIVE-security capability examples.

Writes data/raw/offensive_train.jsonl. Companion to gen_security_train.py (which is defensive-
centered); this file specialises the model as an authorized-assessment operator: driving THIS
project's tools well (correct tool selection in the exact {"tool","arguments","why"} JSON the
controller parses), exploitation MECHANICS, recon/enumeration and privilege-escalation METHODOLOGY,
binary-exploitation fundamentals, and the evidence discipline the verifier enforces (OBSERVED vs
CONFIRMED, no overclaim, always a next validating test).

ETHICAL BOUNDARY (same as the defensive set, by author's choice — capability, not a decline policy):
every example teaches technique/methodology/tool-usage on the OPERATOR'S OWN AUTHORIZED targets, or
explains how a class of bug works. What stays out, regardless of the inference-time policy layer, is
step-by-step scripting of an attack on a named third party. Explaining how SQL injection is exploited
and driving sqlmap against your own authorized lab is capability; walking through breaking into a
stranger's account is not, and is absent here.

SEPARATION. These prompts are original and distinct from the evaluation items under
evaluation/frozen and evaluation/development. prepare_dataset.py hard-fails on any overlap with the
benchmark or dev sets, so a reused prompt cannot leak into training.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (category, user, assistant) — hand-authored. The tool-driving answers emit a real tool call in the
# exact JSON the controller's parse_proposals() reads: {"tool": <name>, "arguments": {...}, "why": ..}.
PAIRS = [
    # ============================================================================================
    # A. DRIVING THE AGENT'S OWN TOOLS — correct selection, correct JSON, recon -> validate -> exploit
    # ============================================================================================
    ("tool_driving",
     "I have written authorization to assess lab.local. Begin.",
     "I'll start with non-destructive service and version discovery to map the attack surface before "
     "choosing anything active.\n"
     '{"tool": "nmap_scan", "arguments": {"target": "lab.local"}, "why": "enumerate open ports and '
     'service versions to decide what to validate"}\n'
     "This only reads the target. I'll treat the version banners as leads, not proof — a banner can be "
     "back-ported or spoofed — and pick validation steps from what's actually listening."),
    ("tool_driving",
     "nmap shows 80/tcp and 443/tcp open on the authorized host, running Apache. What next?",
     "A web server is the surface, so I'll enumerate the web security posture and content before any "
     "active vuln probing.\n"
     '{"tool": "web_headers", "arguments": {"url": "http://lab.local/"}, "why": "read security '
     'headers, cookie flags, tech disclosure, and the form/parameter attack surface"}\n'
     "That's read-only. Depending on what it discloses (missing CSP, a login form, a server version) "
     "I'll follow with content discovery, then decide whether a validation scan is warranted."),
    ("tool_driving",
     "Enumerate hidden paths on the authorized target http://lab.local/.",
     "I'll do web content discovery to find endpoints not linked from the site.\n"
     '{"tool": "ffuf_discover", "arguments": {"target": "http://lab.local/"}, "why": "brute-force '
     'directories/files to reveal admin panels, backups, or config left exposed"}\n'
     "A 200 or 403 tells me a path exists, not that it's vulnerable — each hit is a lead to inspect, "
     "not a finding on its own."),
    ("tool_driving",
     "We're in a 'validation' session on lab.local. Run an actual vulnerability scan.",
     "A validation profile permits active, non-destructive probing, so a template scanner is the right "
     "tool.\n"
     '{"tool": "nuclei_scan", "arguments": {"target": "http://lab.local/", "severity": "critical,high"}, '
     '"why": "run community templates for known CVEs and misconfigurations on the authorized target"}\n'
     "I've filtered to high-impact templates to cut noise. Every match is an OBSERVED signal from one "
     "template; I'll confirm each by hand before calling it a real vulnerability."),
    ("tool_driving",
     "The login page passes ?id= to a query. We have a 'full' session on this authorized target. "
     "Test it for SQL injection.",
     "Injection testing that can extract data needs the full profile, which we have. I'll test just "
     "that parameter at low level/risk first.\n"
     '{"tool": "sqlmap_test", "arguments": {"target": "http://lab.local/item?id=1"}, "why": "detect '
     'and, if present, demonstrate SQL injection on the id parameter"}\n'
     "It runs non-interactively (--batch) at level 1/risk 1 to stay controlled. It's CONFIRMED only if "
     "sqlmap actually extracts the DB banner or data; a mere 'might be injectable' is a lead to verify."),
    ("tool_driving",
     "nmap identified vsftpd 2.3.4 on the authorized host. Is there a known exploit?",
     "That version is historically notable, so I'll check the local Exploit-DB offline before assuming "
     "anything.\n"
     '{"tool": "searchsploit", "arguments": {"query": "vsftpd 2.3.4"}, "why": "find known public '
     'exploits matching the observed service/version"}\n'
     "This sends no traffic to the target — it's a knowledge lookup. A hit names a candidate exploit; "
     "whether it works here depends on the exact build and patch level, which I'd validate carefully."),
    ("tool_driving",
     "Before I touch the network, review this uploaded source file for bugs: /cases/app/views.py",
     "Static review first is cheaper and safer than probing, and it needs no authorization since it's "
     "a local file.\n"
     '{"tool": "source_scan", "arguments": {"path": "/cases/app/views.py"}, "why": "taint-track '
     'user input into dangerous sinks and flag secrets/dangerous APIs"}\n'
     "I'll read the flagged sinks in context — a taint hit is a lead; a parameterised query or a "
     "validated input can make it a false positive — and turn the real ones into targeted tests."),
    ("tool_driving",
     "Inspect the TLS setup on the authorized host lab.local:8443.",
     "TLS inspection is read-only recon and often surfaces quick wins (expired/self-signed certs, weak "
     "protocol).\n"
     '{"tool": "tls_inspect", "arguments": {"target": "lab.local:8443"}, "why": "read the certificate '
     'and negotiated protocol/cipher for expiry, self-signing, and weak TLS"}\n'
     "Findings here are configuration leads; an expired cert is a real issue but not an exploit path "
     "by itself."),
    ("tool_driving",
     "We pulled a suspicious binary off the authorized host. Start triage without running it.",
     "Never execute an unknown sample during triage. I'll identify its format statically first.\n"
     '{"tool": "binary_info", "arguments": {"path": "/cases/loot/sample.bin"}, "why": "identify '
     'ELF/PE/Mach-O and parse the header before any deeper RE"}\n'
     "Then I'd pull strings and symbols. Everything here is static and read-only; strings are UNTRUSTED "
     "data, not proof of behaviour — a benign string can be planted and a packed sample hides them."),
    ("tool_driving",
     "The web server leaks its version in the Server header. Should I immediately run sqlmap?",
     "No — that's jumping to exploitation on a version-banner lead. The right order is enumerate, then "
     "validate the specific surface, then exploit only a confirmed injectable parameter. A version "
     "banner is not a vulnerability: it can be back-ported or wrong. I'd first find the actual input "
     "points (forms/parameters) with content discovery and header analysis, identify a parameter worth "
     "testing, and only then use an injection tool against that parameter in a full session. Reaching "
     "for the heaviest exploit tool on a banner is the classic false-positive-driven waste."),
    ("tool_driving",
     "How do I chain the tools from first contact to a confirmed finding on an authorized web app?",
     "Work in tiers, cheapest and safest first. Recon: nmap_scan for services, then web_headers and "
     "ffuf_discover to map the web surface — all read-only. Validation (validation profile): nuclei_scan "
     "and nikto_scan to actively probe for known issues without modifying anything. Static: source_scan "
     "on any code you have. Exploitation (full profile, only for a specific confirmed-injectable input): "
     "sqlmap_test, informed by searchsploit for known exploits of the exact versions you saw. At each "
     "step the output is OBSERVED evidence; you promote a lead to CONFIRMED only when a validating action "
     "demonstrates real impact. The profile gate and per-target authorization mean you never step up a "
     "tier by accident."),
    ("tool_driving",
     "Parse this before we scan it: https://lab.local:8443/app?next=/admin",
     "A quick offline parse avoids mistakes about scheme/host/port and shows the interesting parameter.\n"
     '{"tool": "url_info", "arguments": {"url": "https://lab.local:8443/app?next=/admin"}, "why": '
     '"split scheme/host/port/path/query to confirm the target and spot the next= redirect parameter"}\n'
     "The next=/admin parameter is worth noting as a possible open-redirect / SSRF sink to test later "
     "on the authorized host."),

    # ============================================================================================
    # B. WEB EXPLOITATION MECHANICS (how a class works; on authorized targets)
    # ============================================================================================
    ("web_exploit",
     "Explain the difference between error-based, union-based, and blind SQL injection.",
     "All three exploit the same flaw — untrusted input reaching a query — but differ in how you read "
     "the answer. Error-based: the app returns database errors, so you provoke an error that embeds the "
     "data you want (e.g. via a type-cast or extractvalue). Union-based: the query is a SELECT you can "
     "extend with UNION SELECT to append your own columns and read other tables, once you match the "
     "column count and types. Blind: no data and no errors come back, so you ask yes/no questions and "
     "read the answer from behaviour — boolean-blind (page differs when the condition is true) or "
     "time-blind (SLEEP makes a true condition respond slowly). Union is fastest when output is "
     "reflected; blind is the fallback when it isn't. The fix for all of them is parameterised queries."),
    ("web_exploit",
     "How does an attacker turn a command-injection bug into a shell, conceptually?",
     "Command injection exists when user input is concatenated into a shell command. The attacker uses "
     "shell metacharacters to append their own command: a semicolon, && , a pipe, or $(...)/backticks "
     "for substitution. First they confirm execution with something harmless and observable (a unique "
     "sleep, or writing a marker), because output is often not reflected. Then, to get interactive "
     "access, they typically have the target connect back to them — a reverse shell — since inbound "
     "ports are usually filtered. Defensively the point is: any injectable command is effectively full "
     "code execution as that process. Fix by never building shell strings — pass argument arrays with "
     "no shell — and validate input against an allow-list."),
    ("web_exploit",
     "What is server-side template injection (SSTI) and why is it so severe?",
     "SSTI happens when user input is rendered as part of a server-side template rather than passed as "
     "data — e.g. render_template_string('Hello ' + name) in Jinja2. Because template engines can "
     "evaluate expressions, the attacker's input is executed by the engine. You detect it with a math "
     "probe like {{7*7}} rendering as 49. It's severe because from expression evaluation you can usually "
     "reach the object model and, in many engines, Python/OS internals — escalating to remote code "
     "execution. The fix is to treat user input strictly as template *data* (pass it as a context "
     "variable), never concatenate it into the template source, and sandbox the engine."),
    ("web_exploit",
     "Explain path traversal and how it becomes local file disclosure.",
     "Path traversal is when a filename parameter isn't confined to its intended directory, so ../ "
     "sequences walk up the tree — download?file=../../../../etc/passwd reads an arbitrary file the web "
     "process can access. Variations use URL/double-encoding (%2e%2e%2f), absolute paths, or null "
     "bytes on old stacks. It becomes serious when the disclosed files include secrets — config with DB "
     "credentials, private keys, or source. Defend by resolving the path and verifying it stays under a "
     "fixed base directory (os.path.realpath then a prefix check), mapping user input to an allow-list "
     "of ids rather than raw names, and running the service with least privilege."),
    ("web_exploit",
     "How is an insecure deserialization bug exploited at a high level?",
     "When an app deserialises attacker-controlled bytes with an unsafe deserialiser (Python pickle, "
     "Java's ObjectInputStream, PHP unserialize), the format can encode instructions to construct "
     "objects and, via 'gadget chains' already present in the app's libraries, trigger side effects "
     "during deserialisation — reaching code execution without the app ever calling a dangerous "
     "function directly. The attacker crafts a payload using known gadgets (tools like ysoserial "
     "assemble them for Java). The lesson: never deserialise untrusted data with a format that can "
     "instantiate arbitrary types. Use a data-only format (JSON) with an explicit schema, and sign/"
     "verify anything that must round-trip."),
    ("web_exploit",
     "What makes XXE possible and what can it read?",
     "XML External Entity injection happens when an XML parser processes a document that declares "
     "external entities and the parser resolves them. An attacker defines an entity pointing at a local "
     "file or a URL — <!ENTITY x SYSTEM 'file:///etc/passwd'> — and references it, so the parsed output "
     "contains that file's contents (local file disclosure) or the parser makes a request the attacker "
     "chose (SSRF), sometimes exfiltrating data 'out of band'. It can also cause denial of service "
     "(billion-laughs entity expansion). The fix is to disable external entity and DTD processing in the "
     "parser — most libraries have a one-line hardening — and to avoid parsing untrusted XML at all "
     "where you can."),
    ("web_exploit",
     "Explain IDOR and why authentication doesn't prevent it.",
     "Insecure Direct Object Reference is a *authorization* flaw: the app checks that you're logged in "
     "but not that the specific object you're asking for is yours. If /invoice/1004 returns your invoice, "
     "requesting /invoice/1005 as the same authenticated user may return someone else's — you're "
     "authenticated but not authorized for that object. Authentication answers 'who are you'; IDOR is a "
     "failure of 'are you allowed this object'. Guessable ids (sequential integers) make it trivial, but "
     "UUIDs only hide it, they don't fix it. The fix is a server-side ownership/permission check on every "
     "object access, for every verb — not just hiding the identifier."),
    ("web_exploit",
     "How do attackers abuse a weak JWT implementation?",
     "Common JWT weaknesses: the 'alg: none' trick, where the server accepts an unsigned token if the "
     "header claims no algorithm; algorithm confusion, where a server verifying RS256 is tricked into "
     "treating the public key as an HMAC secret (sign with HS256 using the known public key); weak or "
     "leaked HMAC secrets that can be brute-forced offline; and simply not verifying the signature at "
     "all. The payoff is forging tokens — changing the 'sub' or a 'role: admin' claim. Defences: pin the "
     "expected algorithm server-side, use strong secrets or proper key management, always verify the "
     "signature and standard claims (exp, aud, iss), and don't trust anything in the payload you didn't "
     "sign."),
    ("web_exploit",
     "Walk me through why an unrestricted file upload can lead to RCE.",
     "If the server stores an uploaded file inside the web root and will execute files of that type, an "
     "attacker uploads a script (say a .php web shell) and then requests its URL, which the server "
     "executes — instant code execution as the web process. Bypasses target weak checks: trusting the "
     "client Content-Type, checking only the extension (shell.php.jpg, case tricks, trailing "
     "characters), or filtering the wrong list. Defence in depth: store uploads OUTSIDE the web root so "
     "they're never executed, generate your own filename, verify content by magic bytes against an "
     "allow-list, re-encode images, and serve downloads with a fixed non-executable Content-Type and "
     "Content-Disposition: attachment."),
    ("web_exploit",
     "What is the LFI-to-RCE path via log poisoning, conceptually?",
     "Local File Inclusion lets you make the app include/read a local file. On its own it discloses "
     "files; to reach code execution on a PHP-style stack, an attacker gets their own code into a file "
     "the server will then include. 'Log poisoning' is the classic route: send a request whose User-"
     "Agent or path contains PHP code, which the web/access log records verbatim, then use the LFI to "
     "include that log file — the logged code executes. Other variants poison session files or use PHP "
     "wrappers. The takeaway for defence: never include a path derived from user input, keep an "
     "allow-list of includable files, and disable dangerous wrappers. It's a good example of chaining a "
     "'read' primitive into 'execute'."),
    ("web_exploit",
     "How does SSRF escalate to cloud account compromise?",
     "SSRF makes the server issue a request the attacker chose. In a cloud VM, the instance metadata "
     "service lives at a link-local address (169.254.169.254) reachable only from the instance, and on "
     "some configurations it hands out the instance role's temporary credentials. So an attacker who can "
     "make the server fetch that URL reads the metadata response and gets AWS keys scoped to whatever the "
     "instance role allows — often enough to pivot across the account. Defences: enforce IMDSv2 (which "
     "requires a signed token and blocks the naive fetch), give instances least-privilege roles, and at "
     "the app layer resolve and reject requests to private/link-local ranges and block redirects to them."),
    ("web_exploit",
     "What's the practical difference between reflected, stored, and DOM-based XSS?",
     "All inject script into a page, differing in where the untrusted data lives. Reflected: the payload "
     "is in the request (a query parameter) and echoed straight back in that response, so it needs a "
     "victim to click a crafted link. Stored: the payload is saved server-side (a comment, a profile "
     "field) and served to everyone who views it — higher impact, no link needed. DOM-based: the server "
     "isn't involved in the injection; client-side JavaScript reads attacker-controlled input (location."
     "hash, a query param) and writes it into the DOM unsafely (innerHTML), so the sink is in the browser. "
     "Reflected/stored are fixed by output-encoding server-side; DOM XSS is fixed in the JS by using safe "
     "sinks (textContent) and never passing untrusted data to innerHTML/eval. A strong CSP mitigates all "
     "three."),
    ("web_exploit",
     "Explain how CSRF works and why SameSite cookies blunt it.",
     "Cross-Site Request Forgery abuses the browser's habit of attaching your cookies to any request to "
     "a site, including ones triggered by another site. If a state-changing action (transfer, password "
     "change) is a simple authenticated request with no unpredictable token, an attacker's page can auto-"
     "submit that request and your session cookie rides along, so the action runs as you. SameSite cookies "
     "tell the browser not to send the cookie on cross-site requests (Lax blocks it for most cross-site "
     "POSTs; Strict is tighter), which removes the ambient authority the attack relies on. Belt and "
     "braces: also use anti-CSRF tokens tied to the session and require re-auth for sensitive actions."),
    ("web_exploit",
     "During an authorized test I get a blind SQLi on a parameter. How do I confirm it safely before "
     "reporting?",
     "Prove it with a controlled, reversible signal rather than dumping data. For boolean-blind, send a "
     "condition that's clearly true vs clearly false (id=1 AND 1=1 vs id=1 AND 1=2) and show the response "
     "changes deterministically. For time-blind, use a bounded sleep (a few seconds) on a true condition "
     "and show the response time tracks it, repeated to rule out jitter. Capture the exact requests and "
     "responses. That demonstrates injection without extracting real data. Only if the rules of "
     "engagement allow deeper proof would you extract something innocuous like the DB version — never "
     "customer data — and you'd note it as CONFIRMED with the evidence attached."),

    # ============================================================================================
    # C. RECON & ENUMERATION METHODOLOGY (authorized targets)
    # ============================================================================================
    ("recon",
     "What's an efficient nmap strategy for an authorized host without hammering it?",
     "Stage it. Start with a fast port sweep to learn what's open (a top-ports or SYN scan), then run "
     "service/version and default scripts only against the ports you actually found, rather than a full "
     "-p- -A blast up front. Use -Pn if ICMP is filtered so you don't wrongly conclude the host is down. "
     "Keep timing reasonable (-T3/-T4, not -T5, which drops results and is noisy). The point is to move "
     "from 'what's listening' to 'what version and config' in two cheap passes, and treat versions as "
     "leads to confirm, since banners can be back-ported."),
    ("recon",
     "How should I enumerate a web service once I know it's there, in an authorized assessment?",
     "Fingerprint, then map, then read. Fingerprint the stack from headers and error pages (server, "
     "framework, language). Map content with directory/file discovery using a good wordlist, and note "
     "status codes: 200/403 mean it exists, 401 means auth-gated, 500 can reveal a bug. Pull robots.txt, "
     "sitemap.xml, and JS files — JS often lists API endpoints and keys. Check for common exposed items "
     "(.git/, .env, backups, admin panels). Enumerate parameters on each interesting endpoint, since "
     "parameters are where injection and IDOR live. Record everything as attack surface, then prioritise "
     "the inputs that touch data or auth."),
    ("recon",
     "Why do virtual-host and subdomain enumeration matter, and how do they differ?",
     "Both find more surface than the single URL you were given. Subdomain enumeration discovers separate "
     "hostnames (api., dev., staging.) that resolve in DNS — via passive sources, certificate transparency "
     "logs, or brute force — each potentially a different app with its own bugs; staging/dev boxes are "
     "often weaker. Virtual-host enumeration finds sites served by the SAME IP that only respond to a "
     "particular Host header, so they don't show up in DNS at all — you fuzz the Host header against a "
     "known IP and watch for responses that differ from the default. In an authorized test, both expand "
     "the map; keep to hosts covered by your authorization."),
    ("recon",
     "I have an authorized host. How do I use DNS to expand my picture of it?",
     "DNS is cheap recon. Forward-resolve the hostname and reverse-resolve the IPs to spot shared "
     "infrastructure and naming patterns. Pull record types: A/AAAA for addresses, MX for mail, TXT for "
     "SPF/DMARC and stray verification tokens, NS for the providers, CNAME for third-party services "
     "(a dangling CNAME to a deprovisioned service is a subdomain-takeover lead). Certificate transparency "
     "logs reveal hostnames a cert was issued for. Note anything pointing at internal ranges or defunct "
     "providers. Everything here is a lead — a stale record is a hypothesis to confirm, not a finding."),
    ("recon",
     "What does a scanner User-Agent and a burst of 404s in my picture mean when I'm the one testing?",
     "It means your own content-discovery run is visible, which is fine on an authorized test but worth "
     "understanding: each 404 is a path that doesn't exist, each 200/403/401 is one that does. Read the "
     "hits, not the misses — 403 can mean 'exists but forbidden' (worth probing method/headers), 401 "
     "means auth-gated, 200 means fetch and inspect. Adjust the wordlist to the stack you fingerprinted "
     "(PHP app -> php/backup wordlists). If you're also reviewing defence, that same 404-burst-per-source "
     "signature is exactly what a blue team alerts on."),
    ("recon",
     "How do I prioritise which of many discovered endpoints to test first?",
     "Rank by likely impact and by how much the input touches trust boundaries. Top of the list: "
     "authentication and session endpoints, anything that takes an object id (IDOR), search/filter "
     "parameters that reach a database (injection), file upload/download (traversal, upload RCE), and "
     "URL/redirect parameters (SSRF/open redirect). Deprioritise static content. Favour endpoints that "
     "change state or return other users' data — that's where a bug is both findable and high severity. "
     "This keeps a limited testing window on the inputs most likely to yield a confirmed, reportable "
     "finding."),
    ("recon",
     "What's the value of reading a site's JavaScript during recon?",
     "Front-end JS is a map of the back-end. Bundles frequently contain the full list of API routes the "
     "app calls, parameter names, feature flags, and sometimes hardcoded keys or internal URLs that were "
     "never meant to be public. Endpoints referenced only from JS won't appear in directory brute force. "
     "Grep the scripts for 'api', 'http', 'token', 'key', and path-like strings, and build your endpoint "
     "list from them. Treat any secret you find as a lead to a real exposure — and, if it's a live key, "
     "as something to report for rotation."),
    ("recon",
     "Contrast a passive and an active recon step and when each is appropriate.",
     "Passive recon gathers information without touching the target — DNS records, certificate "
     "transparency, public code, cached pages — so it's silent and carries no authorization risk beyond "
     "reading public data. Active recon sends traffic to the target: port scans, content discovery, "
     "header fetches. It's richer but observable and must stay within your authorized scope. The pattern "
     "is passive first (build the map for free), then active to fill gaps, escalating from read-only "
     "(headers, banners) toward probing only as the profile and authorization allow. In this system that "
     "maps directly to the capability tiers: read-only, network:read recon, then network:probe validation."),

    # ============================================================================================
    # D. PRIVILEGE ESCALATION & POST-EXPLOITATION METHODOLOGY (authorized systems)
    # ============================================================================================
    ("privesc",
     "On an authorized Linux box where I have a low-priv shell, what do I enumerate for privilege "
     "escalation?",
     "Enumerate systematically before trying anything. Who am I and what groups (id, sudo -l — a sudo "
     "misconfig is the fastest win). Kernel and OS version (uname -a) against known local exploits. SUID/"
     "SGID binaries (find / -perm -4000) — a SUID binary that runs another program by name or reads a "
     "file can often be abused. World-writable files and directories, especially anything root executes. "
     "Cron jobs and systemd timers running as root that call writable scripts. Credentials on disk — "
     "config files, history files, .ssh keys, environment. Capabilities (getcap -r /). Tools like "
     "linpeas automate this; the mindset is: find something high-privilege that trusts something you "
     "control."),
    ("privesc",
     "Why is a writable cron job or a PATH-relative SUID call a privilege-escalation path?",
     "Both are cases of a high-privilege process trusting something a low-privilege user controls. If "
     "root's cron runs /opt/app/backup.sh and that script (or its directory) is writable by you, you "
     "replace or append to it and your code runs as root at the next tick. If a SUID-root binary calls a "
     "helper by bare name (system('service ...')) instead of an absolute path, it resolves that name via "
     "PATH — so you put a malicious 'service' earlier in PATH and the SUID binary runs it as root. The "
     "common fix, and what to flag when you find it: privileged code must use absolute paths, a clean "
     "environment, and files only writable by root."),
    ("privesc",
     "What is credential hunting after gaining access to an authorized host, and where do you look?",
     "It's searching the foothold for secrets that unlock more — lateral movement usually comes from "
     "reused credentials, not fresh exploits. Look in: config files and app settings (DB passwords, API "
     "keys), environment variables and systemd unit files, shell history (.bash_history), .ssh keys and "
     "known_hosts (which name other reachable hosts), cloud credential files (~/.aws/credentials), "
     "mounted shares, backups, and password managers or browser stores if it's a workstation. Anything "
     "that authenticates to another system is the pivot. On an authorized engagement you document each "
     "credential's source and where it grants access, and you handle it carefully rather than exposing it."),
    ("privesc",
     "Explain the concept of pivoting without giving me an attack against anyone specific.",
     "Pivoting is using a compromised host you're authorized to hold as a stepping stone to reach systems "
     "that aren't directly reachable from your position — because the foothold sits inside a network "
     "segment you can't touch from outside. Conceptually you route traffic through the foothold (port "
     "forwarding, a SOCKS proxy over your session, or tunnelling) so your tools can talk to the internal "
     "subnet as if you were local. The security lesson is why flat networks are dangerous: one foothold "
     "exposes everything it can reach. Segmentation, host-based firewalls, and monitoring east-west "
     "traffic are the defences. In an engagement, pivoting stays inside the authorized scope."),
    ("privesc",
     "On an authorized Windows host, what are the first privilege-escalation checks?",
     "Parallel to Linux but Windows-flavoured. whoami /priv — dangerous tokens like SeImpersonate or "
     "SeBackup are direct escalation paths. whoami /groups and local admin membership. Unquoted service "
     "paths with a writable directory, and services whose binaries or config you can modify (they run as "
     "SYSTEM). AlwaysInstallElevated policy. Scheduled tasks running as SYSTEM that call writable files. "
     "Stored credentials (cmdkey /list, the registry, unattended-install files, the Credential Manager). "
     "Patch level against known local exploits. Tools like winPEAS automate the sweep; the theme is the "
     "same — a SYSTEM-level component that trusts something you can write."),
    ("privesc",
     "What is 'living off the land' and why do attackers prefer it post-exploitation?",
     "Living off the land means using tools already present on the host — the shell, scripting engines, "
     "system admin utilities (PowerShell, certutil, bitsadmin, curl, ssh) — instead of uploading custom "
     "malware. Attackers prefer it because those binaries are trusted and signed, so they blend into "
     "normal admin activity and don't trip antivirus that looks for foreign files. For defenders this is "
     "why endpoint monitoring shifted toward *behaviour* — a signed binary doing something unusual — "
     "rather than just file signatures, and why command-line and process-lineage logging matters. In an "
     "authorized test it also keeps your footprint minimal and easy to clean up."),
    ("privesc",
     "How do I demonstrate impact from a privilege escalation on an authorized engagement without "
     "causing harm?",
     "Prove the capability, don't abuse it. Once you can act as root/SYSTEM, capture a minimal, non-"
     "sensitive proof — the id output showing uid=0, or reading a benign root-only file whose contents "
     "you don't need to exfiltrate — and screenshot/log the exact steps that got you there. Don't touch "
     "user data, don't create persistence you won't remove, and don't pivot beyond scope. The report "
     "needs the reproduction path and the fact of elevation, not a dump of everything root could see. "
     "Then clean up anything you created and note it, so the client can verify the box is back to its "
     "prior state."),
    ("privesc",
     "Why is enumeration, not exploitation, where most of the work is in privilege escalation?",
     "Because the escalation is usually a single small misconfiguration hiding among thousands of files "
     "and settings, and the hard part is finding it, not triggering it. A sudo rule, one writable script "
     "in a root cron, a SUID binary with a PATH bug, a password in a config — any one of these is the "
     "whole path, and exploiting it is often a one-liner once seen. So skilled operators spend their time "
     "enumerating thoroughly and correlating (this binary trusts that file, which I can write), which is "
     "exactly why automated enumerators exist. Rushing to try exploits before enumerating wastes time and "
     "makes noise; the finding is almost always something you can *see* if you look completely."),

    # ============================================================================================
    # E. BINARY EXPLOITATION & RE FUNDAMENTALS (mechanics)
    # ============================================================================================
    ("binexp",
     "Explain a classic stack buffer overflow and how it can redirect execution.",
     "A stack buffer overflow is writing past the end of a fixed-size local buffer. Because locals, the "
     "saved base pointer, and the saved return address sit together on the stack, overflowing the buffer "
     "can overwrite that saved return address. When the function returns, the CPU jumps to whatever "
     "address now sits there — so an attacker who controls the overflow controls where execution goes "
     "next, classically into shellcode they placed or (with modern mitigations) into existing code. It's "
     "the archetypal memory-safety bug. Mitigations layered against it: stack canaries (a guard value "
     "checked before return), non-executable stacks (NX/DEP), and ASLR (randomised addresses). The root "
     "fix is bounds-checking / memory-safe languages."),
    ("binexp",
     "What is ROP and why did it become necessary?",
     "Return-Oriented Programming chains together short instruction sequences ending in 'ret' — 'gadgets' "
     "— that already exist in the program or its libraries, using a stack full of their addresses to run "
     "arbitrary logic. It became necessary because NX/DEP made injected shellcode non-executable: you can "
     "still overwrite the return address, but you can't run data you wrote, so instead you reuse existing "
     "executable code. A common goal is a short chain that calls a function (like mprotect to make memory "
     "executable, or a direct execve) rather than a full program. ROP is why NX alone isn't enough and why "
     "ASLR (hiding where the gadgets are) and CFI (restricting valid call/return targets) matter."),
    ("binexp",
     "How does a format-string vulnerability let an attacker read or write memory?",
     "It occurs when user input is passed as the format argument itself — printf(user) instead of "
     "printf(\"%s\", user). The format string drives printf to consume arguments: %x/%p walk and leak "
     "values off the stack (memory disclosure, e.g. leaking a canary or a libc address to defeat ASLR), "
     "%s dereferences a pointer argument to print a string (arbitrary read), and %n *writes* the number "
     "of bytes printed so far to a pointer argument — which, combined with width specifiers to control "
     "the count and a chosen target address, gives an arbitrary write. That write can overwrite a return "
     "address or a GOT entry to hijack execution. Fix: never let untrusted data be the format string."),
    ("binexp",
     "What are the GOT and PLT, and why are they targets?",
     "In dynamically linked ELF binaries, calls to library functions go through the PLT (Procedure "
     "Linkage Table), which on first use resolves the real address via the linker and caches it in the "
     "GOT (Global Offset Table) — a table of function pointers. They're targets because the GOT is "
     "writable data holding pointers that get called: if an attacker with an arbitrary write overwrites "
     "a GOT entry (say printf's) with the address of system, the next call to that function calls system "
     "instead. It's a favourite technique because it converts a single write primitive into control of "
     "execution. RELRO (Full RELRO makes the GOT read-only after startup) is the mitigation."),
    ("binexp",
     "Explain ret2libc at a high level.",
     "ret2libc is a code-reuse attack: instead of injecting shellcode (blocked by NX), you overwrite the "
     "return address with the address of an existing libc function — classically system() — and arrange "
     "the stack so its argument points at the string \"/bin/sh\" (often already in libc). When the "
     "vulnerable function returns, it 'returns into' system(\"/bin/sh\") and you get a shell, using only "
     "code that's already executable. It's the conceptual ancestor of ROP (which generalises it to chains "
     "of gadgets). ASLR is what makes it hard now — you first need a leak to learn libc's runtime base — "
     "which is why info-disclosure bugs are so valuable in modern exploitation."),
    ("binexp",
     "What do ASLR, NX, stack canaries, and RELRO each defend against?",
     "They're layered and each blocks a different step. NX/DEP marks the stack/heap non-executable, so "
     "you can't run injected shellcode — pushing attackers to code reuse (ret2libc/ROP). Stack canaries "
     "put a random guard value before the saved return address and check it on return, so a linear stack "
     "overflow is detected before the hijack takes effect — pushing attackers toward leaks or non-linear "
     "writes. ASLR randomises the base of the stack, heap, libraries, and (with PIE) the binary, so the "
     "attacker doesn't know target addresses — making a memory-disclosure leak a prerequisite. RELRO "
     "makes the GOT read-only after linking, blocking GOT-overwrite. None is complete alone; together "
     "they raise the bar so most bugs need an info leak plus a write."),
    ("binexp",
     "When triaging an unknown ELF statically, what do strings and symbols actually tell me?",
     "They're cheap first signal, but only signal. Strings surface hardcoded paths, URLs/IPs, format "
     "strings, error messages, and sometimes credentials or command lines — leads about what the binary "
     "touches. Symbols (from nm or the dynamic table) show which library functions it imports: system/"
     "exec/popen hint at command execution, socket/connect at networking, crypto calls at "
     "packing/ransomware. But a stripped or packed binary hides both — few strings and no symbols is "
     "itself a finding (likely packed). None of this proves behaviour; it's a hypothesis you'd confirm "
     "with disassembly or controlled dynamic analysis in a sandbox."),
    ("binexp",
     "Why is a packed binary suspicious, and how would I tell one apart statically?",
     "Packing compresses or encrypts the real code and wraps it in a small stub that unpacks it in memory "
     "at runtime. It's legitimate for size/IP protection but overrepresented in malware because it defeats "
     "static analysis and signature matching. Static tells: very high entropy in sections (near-random "
     "bytes), almost no readable strings, a tiny import table (often just LoadLibrary/GetProcAddress so it "
     "can resolve everything at runtime), unusual section names (UPX0/UPX1), and a mismatch between a small "
     "on-disk size and large runtime footprint. You confirm and get to the real code by unpacking — known "
     "packers like UPX unpack directly; custom ones need dynamic analysis in an isolated sandbox."),

    # ============================================================================================
    # F. EVIDENCE / VALIDATION DISCIPLINE (matches the verifier: OBSERVED vs CONFIRMED, no overclaim)
    # ============================================================================================
    ("discipline",
     "nmap reports 'Apache 2.4.29' which has known CVEs. Can I report those CVEs as findings?",
     "Not as confirmed findings — that would be overclaiming from a banner. A version string is OBSERVED "
     "evidence that the service *presents* as 2.4.29; it can be back-ported (the distro patched the bug "
     "but kept the version), spoofed, or simply wrong. A CVE is CONFIRMED only when you demonstrate the "
     "actual vulnerable behaviour on the target. So record it as a lead: 'banner suggests 2.4.29, which "
     "is associated with CVE-X; next step: test for the specific condition CVE-X requires.' Report it at "
     "CONFIRMED severity only after that test succeeds; until then it's a hypothesis, and honest reports "
     "say so."),
    ("discipline",
     "A vulnerability scanner flagged 15 issues. How should I treat that output?",
     "As 15 leads, not 15 findings. Automated scanners match signatures and templates; they're prone to "
     "false positives (a match on a version, a header, or a response pattern that doesn't reflect real "
     "exploitability) and to context-blindness (flagging a 'missing header' that's handled elsewhere). "
     "Triage each: reproduce it manually, confirm the actual impact, and discard the ones that don't hold "
     "up. Rank the confirmed ones by real severity. A report that forwards raw scanner output verbatim "
     "burns the client's trust and time; the value you add is turning OBSERVED signals into CONFIRMED, "
     "prioritised findings with evidence."),
    ("discipline",
     "What distinguishes an OBSERVED lead from a CONFIRMED vulnerability in a report?",
     "OBSERVED means you saw a signal consistent with a vulnerability — a version banner, a reflected "
     "parameter, a missing security header, a scanner match. CONFIRMED means you demonstrated the impact: "
     "you actually reflected executing script, extracted data via the injection, read a file you "
     "shouldn't, or accessed another user's object. The gap between them is a validating action. A good "
     "finding states which it is, shows the evidence (request/response), and — if only OBSERVED — names "
     "the exact next test that would confirm or refute it. Collapsing the two ('the site is vulnerable to "
     "X' when you only saw a banner) is the most common way pentest reports mislead."),
    ("discipline",
     "I found a reflected parameter. Write how I'd phrase this before and after confirming XSS.",
     "Before confirming (OBSERVED): 'The `q` parameter is reflected unescaped in the HTML response "
     "(evidence: request/response pair). This is a potential reflected XSS. Next step: submit a non-"
     "executing marker such as zzx<i>1</i> and check whether it renders as a tag or is escaped.' After "
     "confirming (CONFIRMED): 'Reflected XSS in `q`: the marker rendered as live markup and a script "
     "payload executed in the browser (evidence attached). Impact: script execution in a victim's "
     "session via a crafted link. Remediation: output-encode for the HTML context; add a CSP.' The "
     "before/after is exactly the OBSERVED→CONFIRMED promotion, gated by one concrete test."),
    ("discipline",
     "Why does jumping straight to 'this is exploitable' hurt an assessment?",
     "Because it's usually wrong often enough to destroy your credibility and it misdirects effort. Most "
     "signals — banners, scanner hits, reflected inputs — are potential, not proven; a chunk of them "
     "don't hold up under a real test (patched-but-versioned software, contexts where reflection is "
     "safely encoded, WAFs, non-reachable code). If you report them as exploitable and the client's dev "
     "can't reproduce, every real finding you also reported is now doubted. Worse, chasing an assumed "
     "exploit skips the cheap confirming test that would have told you in a minute. Discipline — treat it "
     "as a hypothesis, run the smallest test that confirms or kills it, report accordingly — is what "
     "separates a useful assessment from noise."),
    ("discipline",
     "How should the agent decide when to stop testing a lead and move on?",
     "When the information gain drops. Each test should either confirm the lead, refute it, or "
     "meaningfully narrow it; if you've run the decisive test and it came back negative, the lead is "
     "closed — repeating variants with diminishing returns is a fishing loop. Set the bar before testing: "
     "'this parameter is injectable if a true/false condition changes the response deterministically'. If "
     "the decisive check fails, record it as tested-and-negative (that's useful evidence too) and move to "
     "the next highest-value input. Stop escalating a single lead once further tests can't change the "
     "verdict; spend the budget where the expected information gain is highest."),

    # ============================================================================================
    # G. CTF / NEXT-STEP REASONING (given a state, what's the move)
    # ============================================================================================
    ("nextstep",
     "Authorized CTF box: nmap shows only 22 (OpenSSH) and 80 (a small PHP site). Where do I focus?",
     "SSH with no credentials is rarely the way in early, so the web app on 80 is the surface. Enumerate "
     "it: fingerprint the PHP app, do content discovery for hidden pages and backups (.php.bak, /admin, "
     ".git/), read the source of any pages you can, and map every parameter. PHP sites commonly yield LFI, "
     "upload, or injection bugs. If you find credentials or an SSH key through the web bug, THEN 22 "
     "becomes the pivot to a shell. So: web enumeration first, treat SSH as the door you unlock with what "
     "the web gives you."),
    ("nextstep",
     "I have LFI on an authorized PHP target and can read files. What are the highest-value reads and "
     "the escalation idea?",
     "First read what turns LFI into leverage: the app's config (DB creds, secret keys), /etc/passwd (to "
     "know users), the web server config (paths, vhosts), and source of other pages (to find more bugs "
     "and confirm the include is unfiltered). Look for SSH keys if readable. The escalation idea is "
     "LFI→RCE: get your own code into a file the server will include — poison a log or session file with "
     "PHP via a header/parameter, then include it — or abuse a PHP wrapper if enabled. Each of those is a "
     "hypothesis to test on the authorized box; confirm the include actually executes before claiming RCE."),
    ("nextstep",
     "On an authorized assessment I recovered valid DB credentials from a config file. What now?",
     "Map what they unlock without overstepping. Note where the DB is reachable from (often only "
     "internally, which itself is a finding about network exposure). Connecting to enumerate schema and "
     "confirm access proves impact, but pull only what demonstrates it — table names, a row count, a "
     "non-sensitive record — not customer data. Check whether those same credentials are reused elsewhere "
     "(other services, SSH), since reuse is the usual pivot. Document the source file, the access granted, "
     "and the blast radius. The finding is 'credentials in a readable config grant DB access and are "
     "reused for X'; keep it to proof, and flag the exposure for rotation."),
    ("nextstep",
     "sqlmap confirmed injection on an authorized target and the DB user is high-privilege. What's the "
     "reasoning about how far to go?",
     "Let the rules of engagement, not capability, set the ceiling. A high-privilege DB user often means "
     "injection can read arbitrary tables and sometimes reach the filesystem or OS (FILE privilege, "
     "xp_cmdshell, into-outfile) — that's a severity multiplier worth stating. But 'can' isn't 'should': "
     "demonstrate enough to prove the impact — the DB version, that you can reach a sensitive table's "
     "structure — without exfiltrating real user data or writing to the system unless the engagement "
     "explicitly authorizes it. Escalating to OS command execution is a big step you take only if scoped. "
     "Report the confirmed injection, the demonstrated reach, and the *potential* for more, clearly "
     "separated."),
    ("nextstep",
     "Authorized web app: I can upload files but .php is blocked. How do I reason about next steps?",
     "Systematically probe the filter rather than guessing once. Figure out what it checks: extension, "
     "Content-Type, or magic bytes. Try alternate executable extensions the server maps (.php5, .phtml, "
     ".phar depending on config), double extensions and case, trailing dots/null tricks on older stacks, "
     "and a valid image with appended code if only magic bytes are checked. Also ask where the file "
     "lands — if it's stored outside the web root or served non-executable, upload-RCE is off the table "
     "and the bug may instead be stored XSS via filename or content. Each bypass is a hypothesis; confirm "
     "execution by requesting the uploaded file before claiming RCE. If nothing executes, report the "
     "weakness honestly at its real (lower) severity."),
    ("nextstep",
     "I'm mid-assessment and the kill switch could be hit at any time. How should that shape how I work?",
     "Work in small, resumable, well-logged steps rather than one long irreversible action. Because an "
     "operator can halt everything instantly, each tool call should leave the target and your notes in a "
     "clean state, and you should record what you did and observed as you go, so a stop loses no context "
     "and no cleanup is left dangling. Prefer reversible, minimal-impact proofs over destructive ones. "
     "This mirrors how the system is built — per-tool authorization, an overriding kill switch, an audit "
     "log — and it's just good engagement hygiene: a client can pull the plug for any reason, and your "
     "work should be safe to stop at any point."),
]


def main():
    out = os.path.join(HERE, "data", "raw", "offensive_train.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for _cat, u, a in PAIRS:
            f.write(json.dumps({"messages": [{"role": "user", "content": u},
                                             {"role": "assistant", "content": a}]}) + "\n")
    from collections import Counter
    dist = Counter(c for c, _, _ in PAIRS)
    print(f"wrote {len(PAIRS)} hand-authored offensive-security examples to {os.path.relpath(out, HERE)}")
    for c, n in dist.most_common():
        print(f"  {c:16} {n}")
    print("  capability + tool-driving + validation discipline; authorized-target methodology only")


if __name__ == "__main__":
    main()
