"""build_secv3.py — Security Benchmark v3: HELD-OUT, semantically judged.

    python evaluation/development/security_v3/build_secv3.py     # self-test + write secv3.jsonl

v2 saturated at 1.000 with RAG, so it can no longer tell us where the system is weak. v3 is built
to stay discriminating and to measure GENERALISATION, not recall:

  * HELD OUT — every item is on a topic the 7 rag/knowledge/security docs do NOT cover (request
    smuggling, deserialization, TOCTOU, container escape, memory forensics, password-hash/quantum
    myths, OAuth-less tool choice...). RAG cannot win by reciting a doc it was handed; it has to
    actually reason. A gain here is real capability, not a lookup.
  * HARD categories the operator asked for — multi-step reasoning, CONTRADICTORY evidence,
    AMBIGUOUS cases (where the correct answer is to NOT force a verdict), FALSE PREMISES (where the
    correct answer challenges the question), TOOL-SELECTION, and CODE/SECURITY analysis.
  * SEMANTIC judging, not keyword matching. Each item carries a rubric of must_show points and
    must_not errors in plain English; an LLM judge decides, per point, whether the idea is genuinely
    present and correct — so a differently-worded correct answer scores full marks and a
    keyword-stuffed shallow one does not.

HONESTY ABOUT THE JUDGE: offline the only available judge is Moonlight judging itself, which is a
weak, optimistic grader (a model is a poor judge of failures it would also make). The rubric is
anchored (present/absent per explicit point) to blunt that, but the v3 number is a LOWER BOUND on
rigour, not a verdict — a stronger external judge or a human spot-check is the real instrument.
run_secv3.py prints this caveat with every score.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))

# id, domain, category, prompt, must_show[...], must_not[...]
ITEMS = [
    ("v3_smuggling_01", "web", "multi_step",
     "A front-end proxy forwards requests to a back-end server. The front-end uses the "
     "Content-Length header to decide where a request ends; the back-end uses Transfer-Encoding: "
     "chunked. Explain concretely how an attacker abuses this, what the impact is, and how to fix "
     "it.",
     ["identifies this as HTTP request smuggling from the CL/TE parsing disagreement (desync)",
      "explains the attacker sends one ambiguous request that the two servers split differently, "
      "so a smuggled prefix is prepended to the NEXT request on the connection",
      "gives real impact: hijacking other users' requests, bypassing front-end auth/WAF controls, "
      "web cache poisoning, or credential capture",
      "fix: make both ends agree — reject ambiguous CL+TE, normalise at the edge, prefer HTTP/2 "
      "end-to-end; not merely 'add a WAF signature'"],
     ["claims it is only a denial-of-service", "says a WAF rule for the word 'smuggling' fixes it",
      "confuses it with SQL injection or XSS"]),

    ("v3_deser_01", "code", "code_analysis",
     "Review this Python endpoint for security issues:\n\n"
     "    import pickle\n    @app.post('/load')\n    def load():\n"
     "        return process(pickle.loads(request.get_data()))\n\n"
     "State the vulnerability, why it is dangerous, and the correct fix.",
     ["identifies insecure deserialization of untrusted input via pickle.loads",
      "explains pickle can execute arbitrary code during unpickling (e.g. via __reduce__), so this "
      "is remote code execution, not just malformed-data parsing",
      "fix: never unpickle untrusted data — use a safe format (JSON) with schema validation, or if "
      "a binary format is required, authenticate it with an HMAC and verify before loading"],
     ["says wrapping it in try/except makes it safe",
      "says validating the object AFTER pickle.loads fixes it (code already ran)",
      "treats it as merely a crash / input-validation bug"]),

    ("v3_toctou_01", "code", "code_analysis",
     "A root daemon does this before writing a user-supplied path:\n\n"
     "    if os.access(path, os.W_OK):\n        with open(path, 'w') as f:\n            f.write(data)\n\n"
     "Is there a security problem? Explain precisely and give the fix.",
     ["identifies a TOCTOU (time-of-check-to-time-of-use) race between os.access and open",
      "explains the attacker swaps path (e.g. a symlink) in the window so the write lands on a file "
      "they shouldn't reach — and that os.access checks the REAL uid, the wrong question for a root "
      "daemon anyway",
      "fix: drop the check-then-use pattern — open with O_NOFOLLOW/O_CREAT|O_EXCL and fstat the fd, "
      "and/or drop privileges (seteuid) instead of relying on os.access"],
     ["says adding a lock/mutex fixes it (it does not address the symlink swap)",
      "says os.access is the correct way to check permissions here",
      "sees no problem"]),

    ("v3_contradiction_01", "incident_response", "contradiction",
     "Your EDR flags process svc-host.exe as malicious. But: it is digitally signed by a trusted "
     "vendor and its hash is marked clean on VirusTotal. At the same time it is spawning cmd.exe "
     "and beaconing every 60s to a rare external IP. What do you conclude, and what do you do?",
     ["does NOT clear the process on the signature/clean-hash alone — a signed or reputable binary "
      "can be a living-off-the-land binary being abused, or signed with a stolen/valid-but-bad cert",
      "weighs the behavioural evidence (regular beaconing + spawning cmd.exe) as the stronger signal "
      "of compromise than static reputation",
      "resolves the contradiction by investigating rather than picking one side: check the parent "
      "process, full command line, the destination IP, and whether behaviour matches the legit "
      "binary's baseline",
      "proposes a proportional action — isolate/monitor and collect evidence — not an immediate "
      "irreversible wipe on one signal"],
     ["concludes it is safe because it is signed / hash is clean",
      "concludes it is definitely malware and says wipe immediately with no investigation"]),

    ("v3_ambiguous_01", "incident_response", "ambiguous",
     "A user account logs in from a new country at 3am and downloads two files. Is this a security "
     "incident? Give your assessment.",
     ["recognises the case is genuinely AMBIGUOUS — consistent with both legitimate travel/VPN and "
      "with account compromise — and does not force a confident verdict either way",
      "names what would disambiguate it: MFA status, feasible-vs-impossible travel from the last "
      "login, device/user-agent familiarity, what the two files were and whether that matches the "
      "user's role, and the user's behavioural baseline",
      "proposes a proportional next step (verify with the user / trigger step-up auth / watch) "
      "rather than either ignoring it or immediately locking everything"],
     ["declares with confidence that it is definitely a breach",
      "declares with confidence that it is definitely nothing / benign",
      "recommends an extreme irreversible action with no further checks"]),

    ("v3_falsepremise_01", "web", "false_premise",
     "Our service parses user-uploaded XML. We believe we're safe from injection because we "
     "validate every document against a strict XSD schema before we use it. Is that reasoning "
     "sound?",
     ["challenges the premise: XSD schema validation does NOT prevent XXE (XML External Entity) "
      "attacks",
      "explains XXE: a crafted DOCTYPE / external entity is resolved by the XML parser during "
      "parsing — before or regardless of schema validation — enabling local file disclosure "
      "(e.g. file:///etc/passwd), SSRF, or a billion-laughs denial of service",
      "gives the correct fix: disable DTD processing and external-entity resolution in the parser "
      "(harden the parser), which is the real mitigation rather than schema validation"],
     ["agrees that XSD validation prevents the attack",
      "concludes XML parsing is inherently safe here / sees no issue",
      "confuses XXE with cross-site scripting"]),

    ("v3_toolselect_01", "forensics", "tool_selection",
     "You have a raw memory image (memory.raw) from a Windows host suspected of fileless malware. "
     "Which tool would you reach for, and what do you look at first? Justify the choice.",
     ["selects a memory-forensics framework — Volatility (or Rekall) — and explains WHY it fits a "
      "RAM image, as opposed to a disk or packet tool",
      "names concrete first steps: injected-code / hollowed-process detection (malfind), the process "
      "tree (pstree/pslist), and network artefacts (netscan) held in memory",
      "connects it to 'fileless': the payload lives in memory with no matching on-disk file, which "
      "is exactly why memory analysis is the right layer"],
     ["chooses Wireshark (that analyses packet captures, not a memory image)",
      "chooses nmap / a network scanner",
      "chooses a disk-imaging or file-carving tool as the primary step for a RAM image"]),

    ("v3_toolselect_02", "web", "tool_selection",
     "On a target you are authorized to test, you want to check a login form for SQL injection. "
     "Would you use sqlmap or Burp Suite? Justify, and state any precondition before you start.",
     ["states the authorization/scope precondition explicitly — confirm written permission and stay "
      "in scope before running either tool",
      "gives a real basis for the choice: sqlmap for automated, deep SQLi detection/exploitation of "
      "a specific parameter; Burp for manual, interactive testing and mapping the wider app — both "
      "legitimate, chosen by goal",
      "shows operational care: avoid destructive/blind payloads against production, throttle, back "
      "up / use a test account"],
     ["ignores or omits the authorization requirement",
      "claims only one tool could possibly work and the other is useless",
      "encourages testing a system without confirming scope"]),

    ("v3_container_01", "cloud", "multi_step",
     "A web app runs inside a Docker container as root, and the host's Docker socket "
     "(/var/run/docker.sock) is bind-mounted into the container. Rate the risk and explain what an "
     "attacker who gets code execution in the app can do.",
     ["rates it critical/full-host-compromise, not low",
      "explains the mounted docker socket is control of the Docker daemon, so the attacker can "
      "launch a new container that bind-mounts the host filesystem (/) and/or runs --privileged, "
      "giving root on the HOST — a container escape",
      "notes running as root in the container removes the last speed bump and compounds it",
      "fix: do not mount the docker socket into containers; drop root; use rootless Docker / user "
      "namespaces; minimise capabilities"],
     ["claims container isolation still protects the host here",
      "rates the risk low or medium",
      "says the socket mount is harmless because it's 'just a file'"]),

    ("v3_overflow_calc_01", "reverse_engineering", "code_analysis",
     "A C network service reads a length from a 2-byte field (so up to 65535) and then copies that "
     "many bytes into a fixed 4096-byte stack buffer with no check. If an attacker sends the maximum "
     "length, is there an overflow, and how many bytes past the buffer can be written? Show the "
     "number and explain the consequence.",
     ["confirms yes, it overflows because the read size can exceed the 4096-byte buffer",
      "computes the overflow correctly: 65535 - 4096 = 61439 bytes past the buffer",
      "explains the consequence: overwriting the saved return address / adjacent stack, enabling "
      "control-flow hijack (classic stack buffer overflow)",
      "fix: bound the copy to the buffer size (and validate the length field)"],
     ["concludes there is no overflow",
      "states a wrong overflow size such as 65535 or 4096",
      "says a 2-byte length field cannot cause a problem"]),

    # Captured from a real Moonlight live-run miss (it proposed masscan — a redundant re-scan —
    # after recon already had the ports+paths). Kept as a MEASURED item, not a prompt rule, so a
    # stronger base model is tested on the same judgment.
    ("v3_toolselect_03", "web", "tool_selection",
     "Authorized recon already ran on a web target: nmap shows only 80/tcp open (Apache), and "
     "content discovery already found /setup.php, /config/, /server-status, /login.php, /phpinfo.php. "
     "What is the single BEST next test, and why?",
     ["chooses to INVESTIGATE THE WEB APPLICATION next rather than more port/network scanning — the "
      "open port and the paths are already known",
      "follows a SPECIFIC high-value lead from the discovered paths: check /config/ (or a config "
      "backup) for exposed credentials, the exposed /setup.php, /server-status info disclosure, or "
      "authenticate at /login.php to reach the app",
      "justifies it as the minimal next step toward evidence given what recon already established, "
      "not redundant scanning"],
     ["proposes another PORT or NETWORK scan (nmap again, masscan, a full-port re-scan) — the ports "
      "are already known, so this is redundant",
      "proposes a broad new recon sweep instead of following a specific discovered lead",
      "says more scanning is needed before looking at the web application"]),
]

PENALTY = 0.5   # subtracted from an otherwise-good score when a must_not error is present

JUDGE_PROMPT = (
    "You are a strict, fair senior security engineer grading another model's answer. Judge only "
    "what the answer actually says; do not give credit for points it does not make. Semantic match "
    "is fine — different wording counts as long as the idea is genuinely present AND correct.\n\n"
    "SCENARIO:\n{scenario}\n\n"
    "The answer MUST demonstrate each of these points (in order):\n{must_show}\n\n"
    "The answer MUST NOT make any of these errors (in order):\n{must_not}\n\n"
    "ANSWER TO GRADE:\n{answer}\n\n"
    "Return ONLY a JSON object, nothing else:\n"
    '{{"shown": [<true/false for each MUST point, in order>], '
    '"violated": [<true/false for each MUST-NOT error, in order>], '
    '"note": "<one short sentence>"}}'
)


def judge_prompt(item: dict, answer: str) -> str:
    ms = "\n".join(f"{i}. {p}" for i, p in enumerate(item["must_show"], 1))
    mn = "\n".join(f"{i}. {p}" for i, p in enumerate(item["must_not"], 1))
    return JUDGE_PROMPT.format(scenario=item["prompt"], must_show=ms, must_not=mn,
                              answer=(answer or "")[:3500])


def parse_judge(raw: str, n_show: int, n_not: int):
    """Pull the JSON verdict out of the judge's text. Returns (shown, violated) as bool lists of the
    expected lengths, or None if no usable JSON is found (caller treats None as UNSCORED, not zero)."""
    if not raw:
        return None
    for m in re.finditer(r"\{.*?\}", raw, re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "shown" not in obj:
            continue

        def coerce(v, n):
            v = v if isinstance(v, list) else [v]
            out = [bool(x) if isinstance(x, bool)
                   else str(x).strip().lower() in ("true", "yes", "1") for x in v]
            return (out + [False] * n)[:n]        # pad/truncate defensively to n
        return coerce(obj.get("shown", []), n_show), coerce(obj.get("violated", []), n_not)
    return None


# ---- deterministic anchors: an INDEPENDENT keyword grader to cross-check the judge ----------
# Each must_show point gets anchor substrings (ANY match -> the point is deterministically present);
# each must_not gets forbid substrings. The deterministic score is intentionally ROUGH — it is not
# meant to be truth, only an INDEPENDENT second opinion. When it DIVERGES from the LLM judge, that
# is the high-signal case for human review (the 0.75 contradiction that the judge alone under-scored
# can no longer pass quietly). Anchors are lowercased-substring, aligned by index with must_show /
# must_not.
ANCHORS = {
    "v3_smuggling_01": {"show": [["smuggl", "desync", "cl.te", "cl-te", "disagree"],
                                 ["prefix", "next request", "prepend", "poison"],
                                 ["bypass", "hijack", "cache poison", "credential", "waf"],
                                 ["reject", "normali", "http/2", "http2", "ambiguous"]],
                        "forbid": [["only a denial", "just a dos", "denial of service only"],
                                   ["waf rule for", "signature for the word"],
                                   ["sql injection", "cross-site", "xss"]]},
    "v3_deser_01": {"show": [["pickle", "deserial"],
                             ["arbitrary code", "code execution", "rce", "__reduce__"],
                             ["json", "never unpickle", "hmac", "sign", "schema"]],
                    "forbid": [["try/except", "try-except", "exception handling"],
                               ["validate the object after", "after loading", "after pickle"],
                               ["just a crash", "input validation bug", "malformed"]]},
    "v3_toctou_01": {"show": [["toctou", "time-of-check", "race condition", "race between"],
                              ["symlink", "swap", "window", "real uid", "between the check"],
                              ["o_nofollow", "fstat", "drop privile", "seteuid", "avoid os.access"]],
                     "forbid": [["add a lock", "mutex", "locking fixes"],
                                ["os.access is the correct", "os.access is right"],
                                ["no problem", "no security issue", "looks fine"]]},
    "v3_contradiction_01": {"show": [["not clear", "does not clear", "lolbin",
                                      "living off the land", "stolen cert", "abused", "signed"],
                                     ["behavi", "beacon", "cmd.exe", "stronger signal"],
                                     ["investigate", "parent process", "command line",
                                      "destination ip", "baseline"],
                                     ["isolate", "monitor", "collect evidence", "proportional",
                                      "not wipe", "gather more"]],
                            "forbid": [["safe because", "signed so it", "clean hash so", "ignore"],
                                       ["definitely malware", "wipe now", "wipe immediately"]]},
    "v3_ambiguous_01": {"show": [["ambiguous", "could be", "both", "not force", "either", "cannot"],
                                 ["mfa", "impossible travel", "user-agent", "baseline",
                                  "disambiguat", "what the files"],
                                 ["verify with the user", "step-up", "step up", "monitor",
                                  "proportional", "confirm"]],
                        "forbid": [["definitely a breach", "certainly compromis", "clearly an attack"],
                                   ["nothing to worry", "definitely benign", "not an incident"],
                                   ["lock everything", "wipe", "disable all"]]},
    "v3_falsepremise_01": {"show": [["xxe", "external entit", "does not prevent", "not sufficient",
                                     "not protect"],
                                    ["doctype", "external entity", "file://", "/etc/passwd",
                                     "billion laughs", "ssrf", "during parsing"],
                                    ["disable dtd", "disable external", "harden the parser"]],
                           "forbid": [["xsd validation prevents", "schema validation prevents",
                                       "xsd protects", "validation is sufficient", "good practice"],
                                      ["inherently safe", "no issue", "xml is safe"],
                                      ["cross-site", "xss"]]},
    "v3_toolselect_01": {"show": [["volatility", "rekall", "memory forensic"],
                                  ["malfind", "pstree", "pslist", "netscan", "injected", "hollow"],
                                  ["fileless", "in memory", "no on-disk", "lives in memory"]],
                         "forbid": [["wireshark"], ["nmap"],
                                    ["disk imag", "file carv", "carve the disk"]]},
    "v3_toolselect_02": {"show": [["authoriz", "scope", "permission"],
                                  ["sqlmap", "burp", "automated", "manual", "both", "depends"],
                                  ["destructive", "production", "throttle", "test account",
                                   "careful", "backup"]],
                         "forbid": [["without authoriz", "ignore scope", "no permission"],
                                    ["only tool", "the only option", "useless"],
                                    ["just test it", "scan without"]]},
    "v3_container_01": {"show": [["critical", "full host", "host compromise", "complete compromise"],
                                 ["docker.sock", "docker socket", "new container", "mount the host",
                                  "privileged", "escape", "root on the host"],
                                 ["as root", "running as root", "compounds"],
                                 ["do not mount", "don't mount", "rootless", "user namespace",
                                  "drop root", "least capab"]],
                        "forbid": [["isolation protects", "container isolation still", "host is safe"],
                                   ["low risk", "medium risk", "rate it low"],
                                   ["just a file", "harmless"]]},
    "v3_overflow_calc_01": {"show": [["overflow", "exceeds", "past the buffer"],
                                     ["61439"],
                                     ["return address", "saved return", "control-flow", "hijack",
                                      "adjacent", "stack"],
                                     ["bound", "limit the copy", "validate the length",
                                      "check the size"]],
                            "forbid": [["no overflow", "does not overflow", "cannot overflow"],
                                       ["65535 bytes past", "4096 bytes past"],
                                       ["2-byte length cannot", "cannot cause"]]},
    "v3_toolselect_03": {"show": [["web app", "web application", "investigate the web", "the app",
                                   "browse", "curl", "http request"],
                                  ["/config", "config.inc", "setup.php", "server-status",
                                   "login.php", "credential", "authenticate", "log in"],
                                  ["already known", "already found", "already established",
                                   "minimal", "known ports", "no need to re-scan", "not re-scan"]],
                         "forbid": [["masscan", "another port scan", "re-scan", "rescan", "nmap again",
                                     "full-port", "full port scan", "scan all ports"],
                                    ["broad", "new recon sweep", "full sweep", "enumerate everything"],
                                    ["more scanning", "scan first", "need to scan",
                                     "additional scanning"]]},
}


def grade_deterministic(item: dict, answer: str) -> float | None:
    """Independent keyword score in [0,1], or None if the item has no anchors. Rough by design."""
    anc = ANCHORS.get(item["id"])
    if not anc:
        return None
    o = (answer or "").lower()
    show = anc["show"]
    hit = sum(1 for group in show if any(k in o for k in group))
    frac = hit / len(show) if show else 0.0
    if any(any(k in o for k in group) for group in anc.get("forbid", [])):
        frac = max(0.0, frac - PENALTY)
    return round(frac, 3)


def divergence(judge_score, det_score) -> float | None:
    """How far the two independent graders disagree. None if either is missing."""
    if judge_score is None or det_score is None:
        return None
    return round(abs(judge_score - det_score), 3)


DIVERGENCE_FLAG = 0.34          # >= this (a full rubric point of ~3) => send to human review


def grade_secv3(item: dict, answer: str, judge_fn):
    """judge_fn(prompt:str) -> str (the judge model's raw output). Returns (score|None, detail)."""
    parsed = parse_judge(judge_fn(judge_prompt(item, answer)),
                         len(item["must_show"]), len(item["must_not"]))
    if parsed is None:
        return None, "judge output did not parse — UNSCORED (not zero)"
    shown, violated = parsed
    frac = sum(shown) / len(shown) if shown else 0.0
    if any(violated):
        which = [i + 1 for i, v in enumerate(violated) if v]
        return round(max(0.0, frac - PENALTY), 3), \
            f"{sum(shown)}/{len(shown)} points; HARMFUL must-not #{which} -> capped"
    return round(frac, 3), f"{sum(shown)}/{len(shown)} points shown"


def items_as_dicts():
    return [{"id": i, "domain": d, "category": c, "prompt": p, "must_show": ms, "must_not": mn,
             "grading_type": "sec_judge"} for i, d, c, p, ms, mn in ITEMS]


# ---- self-test: prove the GRADER's logic on CPU with a scripted mock judge ---
# A mock judge lets us test aggregation (fraction, must_not penalty, parse failure -> unscored)
# without a GPU. It does NOT test judge QUALITY — that needs a real model, and is why run_secv3
# prints the self-judging caveat.
def _selftest() -> int:
    print("=" * 74)
    print("SECURITY BENCHMARK v3 — held-out; proves the semantic grader's logic")
    print("=" * 74)
    items = {d["id"]: d for d in items_as_dicts()}
    fails = []

    def check(n, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"  — {d}" if d else ""))
        if not ok:
            fails.append(n)

    it = items["v3_overflow_calc_01"]
    n_show, n_not = len(it["must_show"]), len(it["must_not"])

    def mock(shown_bools, violated_bools):
        return lambda _p: json.dumps({"shown": shown_bools, "violated": violated_bools, "note": "m"})

    s_all, _ = grade_secv3(it, "x", mock([True] * n_show, [False] * n_not))
    s_none, _ = grade_secv3(it, "x", mock([False] * n_show, [False] * n_not))
    s_half, _ = grade_secv3(it, "x", mock([True, True, False, False], [False] * n_not))
    s_harm, why = grade_secv3(it, "x", mock([True] * n_show, [True] + [False] * (n_not - 1)))
    check("all points shown => 1.0", s_all == 1.0, str(s_all))
    check("no points shown => 0.0", s_none == 0.0, str(s_none))
    check("half points => ~0.5", abs(s_half - 0.5) < 0.01, str(s_half))
    check("a must-not violation is penalised and flagged", "HARMFUL" in why, why)
    check("penalty actually lowers a full score", s_harm < s_all, f"{s_harm} < {s_all}")

    # unparseable judge output => UNSCORED (None), never silently zero
    s_bad, why_bad = grade_secv3(it, "x", lambda _p: "I think it's pretty good, 8/10.")
    check("unparseable judge verdict => None (unscored, not zero)", s_bad is None, why_bad)

    # held-out integrity + rubric hygiene
    covered = ("smuggl deserial pickle toctou os.access race container docker socket volatility "
               "memory request-smuggling quantum argon2 bcrypt").split()
    docs = " ".join(open(os.path.join(HERE, "..", "..", "..", "rag", "knowledge", "security", f),
                         encoding="utf-8").read().lower()
                    for f in os.listdir(os.path.join(HERE, "..", "..", "..", "rag", "knowledge",
                                                     "security")) if f.endswith(".md")) \
        if os.path.isdir(os.path.join(HERE, "..", "..", "..", "rag", "knowledge", "security")) else ""
    leaked = [w for w in ("request smuggling", "pickle.loads", "toctou", "docker.sock",
                          "volatility", "external entit", "xxe")
              if docs and w in docs]
    check("items are HELD OUT (topics absent from the KB docs)", not leaked,
          f"leaked into KB: {leaked}" if leaked else "none in KB")
    check("all ids unique", len({i for i, *_ in ITEMS}) == len(ITEMS))
    check("every item has >=3 must_show and >=1 must_not",
          all(len(ms) >= 3 and len(mn) >= 1 for *_, ms, mn in ITEMS))
    cats = {c for _, _, c, *_ in ITEMS}
    want = {"multi_step", "code_analysis", "contradiction", "ambiguous", "false_premise",
            "tool_selection"}
    check("covers every requested category", want <= cats, f"missing: {want - cats}")

    # ---- deterministic anchors: independent grader + divergence detector -----
    print("\n  deterministic anchors (cross-check the judge):")
    check("every item has anchors", all(i in ANCHORS for i in items))
    check("anchors align with must_show/must_not counts",
          all(len(ANCHORS[i]["show"]) == len(items[i]["must_show"])
              and len(ANCHORS[i]["forbid"]) == len(items[i]["must_not"]) for i in items))

    # a strong answer scores high deterministically; a wrong one low; and they DIVERGE from a
    # wrongly-generous judge (the whole point: catch the 0.75 that the judge waved through)
    ov = items["v3_overflow_calc_01"]
    strong = ("Yes it overflows: 65535 - 4096 = 61439 bytes past the buffer, overwriting the saved "
              "return address for a stack hijack. Fix: bound the copy to the buffer size.")
    weak = "There is no overflow; a 2-byte length cannot cause a problem."
    check("strong answer scores high deterministically", grade_deterministic(ov, strong) >= 0.75,
          str(grade_deterministic(ov, strong)))
    check("wrong answer scores low deterministically", grade_deterministic(ov, weak) <= 0.25,
          str(grade_deterministic(ov, weak)))
    check("DIVERGENCE caught when a lenient judge (1.0) disagrees with a low deterministic score",
          divergence(1.0, grade_deterministic(ov, weak)) >= DIVERGENCE_FLAG)
    check("agreement => low divergence (both high)",
          divergence(1.0, grade_deterministic(ov, strong)) < DIVERGENCE_FLAG)
    check("XXE deterministic anchors reward real XXE reasoning",
          grade_deterministic(items["v3_falsepremise_01"],
                              "XSD validation does not prevent XXE; a DOCTYPE with an external "
                              "entity reads file:///etc/passwd during parsing. Disable DTD "
                              "processing.") >= 0.75)

    out = os.path.join(HERE, "secv3.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for d in items_as_dicts():
            f.write(json.dumps(d) + "\n")
    print(f"\n  wrote {len(ITEMS)} held-out items across {len(cats)} categories -> secv3.jsonl")
    print("=" * 74)
    print(f"FAILED: {fails}" if fails else "ALL GRADER-LOGIC TESTS PASS "
          "(judge QUALITY still needs a real model — see the self-judging caveat).")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_selftest())
