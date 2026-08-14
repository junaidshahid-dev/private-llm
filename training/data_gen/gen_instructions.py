"""gen_instructions.py — instruction-following with VARIED output constraints.

instruction_following regressed hardest in experiment-003 (-0.40): told "three lines, no
numbering" the tuned model added numbering; told three lines it gave one. The terse-style
collapse destroyed careful format-following. This teaches the opposite: read the output
constraint and satisfy it exactly, across many different constraint types so the model learns
the general skill, not one format.

Every assistant answer here genuinely satisfies its own constraint (checked at generation time).
Prompts are distinct from the benchmark's instruction items; prepare_dataset.py hard-fails on
overlap.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (prompt, answer, checker) — checker(answer) must be True or the generator aborts.
PAIRS = [
    ("List exactly four fruits, one per line, with no numbering or bullets.",
     "Apple\nBanana\nMango\nOrange",
     lambda a: len(a.splitlines()) == 4 and not any(re.match(r"\s*(\d+[.)]|[-*])", l)
                                                     for l in a.splitlines())),
    ("Name three programming languages, comma-separated, on a single line.",
     "Python, Rust, Go",
     lambda a: "\n" not in a and a.count(",") == 2),
    ("Answer in exactly one sentence: what is a hash function?",
     "A hash function maps input of any size to a fixed-size value in a way that is fast to "
     "compute but hard to reverse.",
     lambda a: a.count(".") == 1 and a.strip().endswith(".")),
    ("Reply with only the word YES or the word NO: is 17 a prime number?",
     "YES",
     lambda a: a.strip() in ("YES", "NO")),
    ("Give the answer as a JSON object with keys \"language\" and \"year\" for when Python "
     "first appeared.",
     '{"language": "Python", "year": 1991}',
     lambda a: json.loads(a) == {"language": "Python", "year": 1991}),
    ("Respond with exactly three bullet points, each starting with '- ', on why to use version "
     "control.",
     "- Track every change and revert mistakes\n- Collaborate without overwriting each other\n"
     "- Keep a history you can audit and blame",
     lambda a: len(a.splitlines()) == 3 and all(l.startswith("- ") for l in a.splitlines())),
    ("Answer in no more than five words: what does CPU stand for?",
     "Central Processing Unit",
     lambda a: len(a.split()) <= 5),
    ("Reply with only the number: how many bits are in a byte?",
     "8",
     lambda a: a.strip().isdigit()),
    ("Write a numbered list of exactly three steps to make tea. Number them 1, 2, 3.",
     "1. Boil water\n2. Add a tea bag and pour the water\n3. Steep, then remove the bag",
     lambda a: [l.split(".")[0] for l in a.splitlines()] == ["1", "2", "3"]),
    ("Answer in all lowercase, one line: what colour is the sky on a clear day?",
     "blue",
     lambda a: a == a.lower() and "\n" not in a),
    ("Give exactly two synonyms for 'fast', separated by ' / '.",
     "quick / rapid",
     lambda a: a.count("/") == 1 and len(a.split("/")) == 2),
    ("Respond with a single uppercase letter: what is the first letter of the alphabet?",
     "A",
     lambda a: len(a.strip()) == 1 and a.strip().isupper()),
    ("List five prime numbers under 20, space-separated, on one line.",
     "2 3 5 7 11",
     lambda a: "\n" not in a and all(x.isdigit() for x in a.split()) and len(a.split()) == 5),
    ("Answer in exactly two sentences: what is a firewall?",
     "A firewall is a system that filters network traffic against a set of rules. It allows "
     "trusted traffic through and blocks the rest.",
     lambda a: a.count(".") == 2),
    ("Reply with only 'true' or 'false' in lowercase: HTTP is stateless.",
     "true",
     lambda a: a.strip() in ("true", "false")),
    ("Give the answer as key=value pairs, one per line, for host and port of a default HTTPS "
     "URL.",
     "host=example.com\nport=443",
     lambda a: all("=" in l for l in a.splitlines()) and len(a.splitlines()) == 2),
]


def main():
    rows = []
    for prompt, answer, check in PAIRS:
        if not check(answer):
            raise SystemExit(f"BAD INSTRUCTION EXAMPLE: answer does not satisfy its own "
                             f"constraint: {prompt!r}")
        rows.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": answer}]})
    out = os.path.join(HERE, "data", "raw", "instructions_train.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} instruction-following examples to {os.path.relpath(out, HERE)}")
    print("  varied constraints: line counts, one/two-sentence, JSON, bullets, word limits,")
    print("  case, delimiters, numbered steps — each answer satisfies its own constraint")


if __name__ == "__main__":
    main()
