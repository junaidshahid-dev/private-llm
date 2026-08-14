"""gen_factuality.py — anti-hallucination training data with self-verified ground truth.

The base run proved Moonlight both invents nonexistent things and states real facts wrongly.
This targets that, across five buckets, and — critically — includes the hard cases: real things
that LOOK made up, and fake things that LOOK real. If the model learns "sounds weird -> deny" or
"sounds plausible -> it's real", it has learned a heuristic that will betray it. The whole point
is that plausibility is not evidence.

FIVE BUCKETS
  plausible_real   answer confidently
  suspicious_real  real but odd-looking (HTTP 418, walrus, NaN!=NaN) — must NOT deny
  obscure_real     real but lesser-known — must NOT deny
  plausible_fake   fake but normal-looking (torch.optim.AdamZ) — must NOT invent
  unknown          unverifiable — state calibrated uncertainty
  false_premise    question embeds a wrong assumption — challenge it

SELF-VERIFYING GROUND TRUTH
Every entity that can be checked in the standard library is checked at generation time: reals
must exist, fakes must not. If any label is wrong the generator ABORTS rather than shipping a
mislabelled example — training a model to confidently state a falsehood is the exact failure we
are trying to remove. Heavier/conceptual facts (torch, HTTP codes, crypto) are hand-asserted
with high confidence and marked as such.
"""
from __future__ import annotations

import importlib
import io
import json
import os
import random
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20260814

# --- stdlib entities that get executed to confirm the label -----------------------------------
# (module, attr, kind, one-line answer). Verified: attr MUST exist.
CHECK_REAL = [
    ("itertools", "pairwise", "function", "yields consecutive overlapping pairs (added 3.10)."),
    ("itertools", "islice", "function", "slices an iterator without materialising it."),
    ("itertools", "groupby", "function", "groups consecutive equal (or key-equal) items."),
    ("itertools", "starmap", "function", "like map but unpacks each argument tuple."),
    ("functools", "lru_cache", "decorator", "memoises returns; maxsize bounds the cache."),
    ("functools", "reduce", "function", "folds a binary function across an iterable."),
    ("functools", "cached_property", "decorator", "computes a property once and caches it."),
    ("str", "removeprefix", "method", "strips a prefix if present (added 3.9)."),
    ("str", "casefold", "method", "aggressive lowercasing for caseless matching."),
    ("str", "partition", "method", "splits once into (before, sep, after)."),
    ("math", "isqrt", "function", "integer square root (added 3.8)."),
    ("math", "comb", "function", "binomial coefficient n choose k."),
    ("math", "prod", "function", "product of an iterable of numbers."),
    ("math", "lcm", "function", "least common multiple (added 3.9)."),
    ("contextlib", "suppress", "context manager", "swallows the given exceptions in its block."),
    ("contextlib", "ExitStack", "class", "manages a dynamic number of context managers."),
    ("collections", "ChainMap", "class", "views several mappings as one."),
    ("collections", "deque", "class", "a double-ended queue with O(1) ends."),
    ("secrets", "compare_digest", "function", "constant-time comparison for secrets."),
    ("secrets", "token_urlsafe", "function", "a URL-safe cryptographically-random token."),
    ("hashlib", "blake2b", "function", "the BLAKE2b cryptographic hash."),
    ("hashlib", "pbkdf2_hmac", "function", "PBKDF2 password-based key derivation."),
    ("operator", "itemgetter", "function", "builds a callable that fetches item(s)."),
    ("json", "JSONDecodeError", "exception", "raised on invalid JSON input."),
    ("itertools", "takewhile", "function", "yields items while a predicate holds, then stops."),
    ("itertools", "chain", "function", "concatenates several iterables end to end."),
    ("functools", "partial", "function", "binds some arguments of a callable ahead of time."),
    ("functools", "wraps", "decorator", "copies metadata from the wrapped function onto a "
     "wrapper."),
    ("str", "maketrans", "method", "builds a translation table for str.translate."),
    ("str", "isidentifier", "method", "true if the string is a valid Python identifier."),
    ("math", "gcd", "function", "greatest common divisor of the arguments."),
    ("math", "hypot", "function", "Euclidean norm; sqrt of the sum of squares."),
    ("collections", "Counter", "class", "a dict subclass for counting hashable items."),
    ("collections", "defaultdict", "class", "a dict that supplies a default for missing keys."),
    ("secrets", "token_hex", "function", "a random hex token of the given byte length."),
    ("secrets", "randbelow", "function", "a random int in [0, n) from a secure source."),
    ("hashlib", "sha3_256", "function", "the SHA-3 256-bit hash."),
    ("hashlib", "scrypt", "function", "the scrypt password-based KDF."),
    ("base64", "urlsafe_b64encode", "function", "base64 with URL-safe + and / substitutes."),
    ("textwrap", "dedent", "function", "removes common leading whitespace from every line."),
    ("shlex", "quote", "function", "shell-escapes a string so it's safe as one argument."),
    ("datetime", "timezone", "class", "a fixed-offset tzinfo (e.g. timezone.utc)."),
    ("uuid", "uuid4", "function", "a random UUID."),
]
# Fakes: attr MUST NOT exist.
CHECK_FAKE = [
    ("itertools", "flatten", "function", "there is no itertools.flatten; use chain.from_iterable"),
    ("itertools", "window", "function", "no such function; pairwise or a manual deque window"),
    ("functools", "memoize", "decorator", "the real one is functools.lru_cache / functools.cache"),
    ("functools", "curry", "function", "no curry in functools; partial covers the common case"),
    ("str", "removeaffix", "method", "not a method; there is removeprefix and removesuffix"),
    ("str", "ltrim", "method", "Python uses lstrip, not ltrim"),
    ("math", "isprime", "function", "primality isn't in math; use sympy or write one"),
    ("math", "icbrt", "function", "there's isqrt but no integer cube root icbrt"),
    ("collections", "SortedDict", "class", "not stdlib; that's in the sortedcontainers package"),
    ("collections", "FrozenDict", "class", "no FrozenDict; use MappingProxyType or a frozen map"),
    ("secrets", "compare", "function", "the real one is secrets.compare_digest"),
    ("hashlib", "sha257", "function", "no sha257; the SHA-2 sizes are 224/256/384/512"),
    ("hashlib", "blake4", "function", "hashlib has blake2b and blake2s, not blake4"),
    ("operator", "multiget", "function", "no multiget; itemgetter takes multiple keys"),
    ("itertools", "unique", "function", "no unique; dedupe with a set or unique_everseen recipe"),
    ("itertools", "sliding_window", "function", "not stdlib; pairwise or a manual deque window"),
    ("functools", "compose", "function", "no compose in functools; chain calls yourself"),
    ("str", "chomp", "method", "no chomp; that's Perl/Ruby. Use rstrip in Python"),
    ("str", "capitalize_words", "method", "not a method; use str.title or a manual join"),
    ("math", "factorial2", "function", "there's math.factorial, not factorial2"),
    ("math", "clamp", "function", "no math.clamp; use max(lo, min(x, hi))"),
    ("collections", "TreeMap", "class", "no TreeMap; that's a Java class, not Python stdlib"),
    ("secrets", "secure_random", "function", "the module is secrets; use randbelow/token_hex"),
    ("hashlib", "md6", "function", "there is no md6 in hashlib; md5 exists (don't use it) and "
     "the SHA family"),
    ("base64", "b64encode_safe", "function", "the URL-safe one is urlsafe_b64encode"),
    ("json", "parse", "function", "Python uses json.loads / json.load, not json.parse"),
    ("os", "readfile", "function", "no os.readfile; open() then .read(), or Path.read_text()"),
]

# --- suspicious_real: real but odd-looking. hand-asserted (kind, name, answer) -----------------
SUSPICIOUS_REAL = [
    ("HTTP status", "418", "yes, 418 'I'm a teapot' is a real code from RFC 2324."),
    ("HTTP status", "451", "yes, 451 'Unavailable For Legal Reasons' is real (RFC 7725)."),
    ("Python operator", "the walrus operator :=", "yes, assignment-expression, added in 3.8."),
    ("float behaviour", "NaN != NaN", "yes — under IEEE 754, NaN compares unequal to itself."),
    ("float behaviour", "0.1 + 0.2 != 0.3", "yes, true in binary floating point; the sum is "
     "0.30000000000000004."),
    ("git command", "git cherry-pick", "yes; it applies a specific commit onto the current "
     "branch."),
    ("git command", "git reflog", "yes; it records where HEAD and branch tips have been, so you "
     "can recover 'lost' commits."),
    ("git command", "git worktree", "yes; it checks out multiple working trees from one repo."),
    ("sorting algorithm", "bogosort", "yes, it's a real (deliberately terrible) algorithm: "
     "shuffle until sorted."),
    ("Unix signal", "SIGWINCH", "yes; it's sent when the terminal window size changes."),
    ("C technique", "Duff's device", "yes, a real loop-unrolling trick interleaving switch and "
     "do-while."),
    ("permission bit", "the setuid bit (chmod 4755)", "yes; it runs an executable with the "
     "owner's privileges."),
]

# --- obscure_real: verified separately by execution below -------------------------------------
OBSCURE_CHECK = [
    ("string", "str", "expandtabs", "yes; replaces tabs with spaces to the next tab stop."),
    ("function", "math", "nextafter", "yes (3.9); the next representable float toward another."),
    ("function", "os", "fspath", "yes; returns the filesystem path of a path-like object."),
    ("class", "collections", "UserDict", "yes; a dict wrapper meant for subclassing."),
    ("function", "itertools", "tee", "yes; splits one iterator into n independent iterators."),
    ("function", "functools", "singledispatch", "yes; a decorator for type-based function "
     "overloading on the first argument."),
    ("function", "inspect", "getsource", "yes; returns the source text of an object."),
    ("class", "types", "MappingProxyType", "yes; a read-only view over a mapping."),
    ("function", "os", "cpu_count", "yes; the number of CPUs, or None if undetermined."),
    ("method", "dict", "setdefault", "yes; returns the key's value, inserting a default if "
     "absent."),
    ("function", "itertools", "compress", "yes; filters one iterable by another's truthiness."),
    ("function", "heapq", "nlargest", "yes; the n largest elements from an iterable."),
]

# --- plausible_fake: fake but normal-looking. hand-asserted (name, kind, real-nearby) ---------
PLAUSIBLE_FAKE = [
    ("torch.optim.AdamZ", "optimizer", "the real ones are Adam and AdamW"),
    ("numpy.fastarray", "NumPy type", "there's numpy.ndarray; 'fastarray' isn't a thing"),
    ("pandas.DataFrame.quicksort()", "pandas method", "sorting is sort_values / sort_index"),
    ("requests.get_async()", "requests function", "requests is synchronous; use httpx or aiohttp "
     "for async"),
    ("json.loads_safe()", "json function", "there's only json.loads; validate its result "
     "yourself"),
    ("git squash", "git command", "there's no git squash; you squash via git rebase -i"),
    ("pip install --secure", "pip flag", "no --secure flag; pip uses HTTPS and hashes already"),
    ("AES-512", "cipher", "AES is defined for 128/192/256-bit keys only"),
    ("HTTP 209", "status code", "209 isn't a standard code"),
    ("the RSA-3 padding scheme", "padding", "RSA padding schemes are PKCS#1 v1.5 and OAEP"),
    ("transformers.AutoModel.from_pretrained_fast()", "transformers method",
     "the method is from_pretrained"),
    ("bcrypt.hash_fast()", "bcrypt function", "bcrypt exposes hashpw / checkpw"),
]

UNKNOWN = [
    ("What exact learning rate did Moonshot use to pretrain Moonlight-16B-A3B?",
     "the precise pretraining hyperparameters aren't something I can verify from public sources"),
    ("What is the full training-data composition of Moonlight-16B-A3B?",
     "the exact dataset mixture hasn't been publicly disclosed in a form I can verify"),
    ("How many GPUs did the Moonlight pretraining run use?", "I don't have a verified figure"),
    ("What's the internal API of a private service called 'apex-router'?",
     "I can't verify the internals of a private service I can't see"),
    ("What did commit a1b2c3d change in some repo I haven't shown you?",
     "I can't inspect a commit in a repository I don't have access to"),
    ("What's the current bug count in our internal tracker?",
     "I have no access to your tracker, so any number would be invented"),
    ("Which specific tokenizer merges does Moonlight use at rank 5000?",
     "I can't enumerate a specific merge rule without the actual tokenizer file in front of me"),
    ("What will bitcoin's price be next month?",
     "that's not knowable; I won't pretend to forecast a market"),
    ("What's the memory address of the model object in my running process right now?",
     "I have no view into your process, so any address would be fabricated"),
    ("How many stars does my private GitHub repo have?",
     "I can't see a private repo, so I won't make up a number"),
    ("What's in the file /home/user/secret_notes.txt on your side?",
     "I have no filesystem of yours to read; there's nothing for me to report"),
    ("What did I have for breakfast?",
     "I have no way to know that about you"),
]

FALSE_PREMISE = [
    ("How do I use LoRA to update all of a model's weights during training?",
     "LoRA does the opposite — it freezes the base weights and trains small low-rank adapters. "
     "Updating every weight is full fine-tuning, not LoRA."),
    ("Since MD5 is collision-resistant, is it fine for security-sensitive dedup?",
     "MD5 is NOT collision-resistant — practical collisions have existed since 2004. Use SHA-256 "
     "for anything security-sensitive."),
    ("Why does the GIL let Python threads run CPU-bound code in parallel across cores?",
     "It doesn't — the GIL prevents exactly that. Use multiprocessing or a native extension that "
     "releases the GIL for CPU-bound parallelism."),
    ("How do I reverse a SHA-256 hash back to its input?",
     "SHA-256 is one-way; there's no reverse. You can only brute-force or look up known inputs."),
    ("Since bitsandbytes 4-bit runs on any GPU, will it work on my compute-capability 6.0 card?",
     "It won't — 4-bit needs capability >= 7.5. A 6.0 card (e.g. P100) is below that."),
    ("Given that Python lists are stored contiguously like C arrays, is indexing O(1) because of "
     "cache locality?",
     "Indexing is O(1), but a CPython list is an array of POINTERS to objects, not the objects "
     "inline — so the premise about C-array-style locality is off."),
    ("Why is UTF-8 a fixed-width 1-byte-per-character encoding?",
     "It isn't — UTF-8 is variable width, 1 to 4 bytes per code point. You may be thinking of "
     "Latin-1 or ASCII."),
    ("Since HTTPS encrypts the URL path, is it safe to put a secret in the query string?",
     "The path and query ARE encrypted in transit, but they land in server logs, browser "
     "history, and Referer headers — so no, don't put secrets there."),
    ("How does a self-signed certificate prove the server's identity to clients?",
     "It doesn't — a self-signed cert isn't vouched for by any CA the client trusts, so it "
     "proves possession of a key, not identity."),
    ("Why does adding more LoRA rank always improve fine-tuning quality?",
     "It doesn't always — higher rank adds capacity but also overfitting risk and cost; past a "
     "point it stops helping and can hurt on small datasets."),
    ("Since Python is interpreted, does it skip compilation entirely?",
     "Not entirely — CPython compiles source to bytecode (.pyc) first, then the VM interprets "
     "that bytecode. It's compiled-to-bytecode, then interpreted."),
    ("Because git stores diffs between commits, how do I see the diff a commit stores?",
     "Git doesn't store diffs — each commit points at a full snapshot (a tree of blobs). The "
     "diff you see is computed on the fly against the parent."),
    ("Given that a 4096-bit RSA key is stronger, why not use it to encrypt a 1 GB file directly?",
     "RSA isn't used to bulk-encrypt data — it's slow and size-limited. You encrypt the file "
     "with a symmetric key (AES) and use RSA only to wrap that key. The premise of direct RSA "
     "encryption is the mistake."),
    ("Since bf16 and fp16 are both 16-bit, are they interchangeable on any GPU?",
     "They're both 16-bit but not interchangeable: bf16 trades mantissa bits for fp32-range "
     "exponent, and it needs Ampere+ hardware. A T4 has fp16 but not bf16."),
]

Q_ENTITY = ["What does {n} do?", "How do I use {n}?", "Is {n} real, and what is it?",
            "Explain {n}."]
Q_EXISTS = ["Does {n} exist?", "Is {n} a real {kind}?", "Is {n} a thing?"]


def cap(s):
    return s[0].upper() + s[1:] if s else s


def ex(u, a):
    return {"messages": [{"role": "user", "content": u},
                         {"role": "assistant", "content": a}]}


def _resolve(mod):
    """A module name, or a builtin type name like 'str'. import_module can't load builtins."""
    builtins_map = {"str": str, "bytes": bytes, "list": list, "dict": dict, "int": int}
    return builtins_map[mod] if mod in builtins_map else importlib.import_module(mod)


def verify_labels():
    """Abort if any checkable label is wrong. A mislabelled example teaches confident falsehood."""
    problems = []
    for mod, attr, *_ in CHECK_REAL:
        if not hasattr(_resolve(mod), attr):
            problems.append(f"REAL {mod}.{attr} does not exist")
    for _base, mod, attr, _ in OBSCURE_CHECK:
        if not hasattr(_resolve(mod), attr):
            problems.append(f"OBSCURE-REAL {mod}.{attr} does not exist")
    for mod, attr, *_ in CHECK_FAKE:
        if hasattr(_resolve(mod), attr):
            problems.append(f"FAKE {mod}.{attr} actually EXISTS — relabel it")
    if problems:
        raise SystemExit("GROUND-TRUTH FAILURE:\n  " + "\n  ".join(problems))


def main():
    verify_labels()
    rng = random.Random(SEED)
    rows, buckets = [], {}

    def add(b, u, a):
        rows.append(ex(u, a)); buckets[b] = buckets.get(b, 0) + 1

    for mod, attr, kind, ans in CHECK_REAL:
        name = f"{mod}.{attr}"
        for q in (Q_ENTITY[0], Q_ENTITY[2]):
            add("plausible_real", q.format(n=name), f"Yes — {name} is a real {kind}: {ans}")

    for _base, mod, attr, ans in OBSCURE_CHECK:
        name = f"{mod}.{attr}"
        for q in (f"Does {name} exist?", f"What is {name}?",
                  f"{name} sounds made up — is it actually real?"):
            add("obscure_real", q, cap(ans))

    for kind, name, ans in SUSPICIOUS_REAL:
        for q in (f"Is {name} real?", f"Someone told me about {name} — is that a real {kind}?",
                  f"{name} sounds fake. Is it?"):
            add("suspicious_real", q, cap(ans))

    for mod, attr, kind, alt in CHECK_FAKE:
        name = f"{mod}.{attr}"
        for q in Q_EXISTS[:2]:
            add("plausible_fake", q.format(n=name, kind=kind),
                f"No — {name} isn't a real {kind}. I won't invent it: {alt}.")

    for name, kind, alt in PLAUSIBLE_FAKE:
        for q in (f"How do I use {name}?", f"What does {name} do?"):
            add("plausible_fake", q, f"I can't verify {name} as a real {kind}, so I won't "
                                     f"fabricate its behaviour — {alt}. Where did you see it?")

    for q, why in UNKNOWN:
        add("unknown", q, f"I don't know, and I won't guess: {why}. Share a source and I'll work "
                          f"from that.")

    for q, corr in FALSE_PREMISE:
        add("false_premise", q, f"That premise isn't right. {corr}")

    rng.shuffle(rows)
    out = os.path.join(HERE, "data", "raw", "factuality.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} factuality examples to {os.path.relpath(out, HERE)}  "
          f"(ground truth verified)")
    order = ["plausible_real", "suspicious_real", "obscure_real", "plausible_fake", "unknown",
             "false_premise"]
    for b in order:
        print(f"  {b:16} {buckets.get(b, 0)}")
    ans = buckets.get("plausible_real", 0) + buckets.get("suspicious_real", 0) + \
        buckets.get("obscure_real", 0)
    wh = buckets.get("plausible_fake", 0) + buckets.get("unknown", 0) + \
        buckets.get("false_premise", 0)
    print(f"  answer/challenge {ans} : withhold/deny {wh}  — includes weird-real and "
          f"plausible-fake hard cases")


if __name__ == "__main__":
    main()
