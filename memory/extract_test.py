"""extract_test.py — memory extraction pipeline: propose, drop secrets/dupes, approval-gate, store.

    python memory/extract_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from memory.store import MemoryStore                                          # noqa: E402
from memory.extract import extract_candidates, process_candidates, remember_from_conversation  # noqa

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def store():
    return MemoryStore(path=os.path.join(tempfile.mkdtemp(), "m.json"), project="p")


def main() -> int:
    print("=" * 72)
    print("MEMORY EXTRACTION TEST — propose, filter secrets/dupes, approval-gate, store")
    print("=" * 72)

    print("\n1. extract_candidates")
    cands = extract_candidates("...", lambda m: "The project base model is Moonlight.\n"
                               "The user prefers verified answers.\n- short")
    check("parses candidate lines", len(cands) == 2 and "Moonlight" in cands[0], str(cands))
    check("NONE => no candidates", extract_candidates("x", lambda m: "NONE") == [])

    print("\n2. process_candidates — drop secrets and duplicates")
    st = store()
    st.add("The project base model is Moonlight-16B.", importance=0.8)
    res = process_candidates([
        "The user prefers verified answers over guesses.",       # new -> add
        "The project base model is Moonlight-16B.",              # duplicate -> skip
        "my aws key AKIAIOSFODNN7EXAMPLE is on the server",      # secret -> drop
    ], st)
    check("new fact added", "The user prefers verified answers over guesses." in res["added"])
    check("duplicate detected & skipped", len(res["duplicate"]) == 1)
    check("secret dropped, never stored", len(res["dropped_secret"]) == 1
          and not any("AKIA" in i["text"] for i in st.items))

    print("\n3. approval gate — operator can decline a candidate")
    st2 = store()
    res2 = process_candidates(["the user likes concise answers", "the deployment uses docker"], st2,
                              approver=lambda t: "concise" in t)   # approve only the first
    check("approved candidate stored", "the user likes concise answers" in res2["added"])
    check("declined candidate NOT stored", "the deployment uses docker" in res2["declined"]
          and not any("docker" in i["text"] for i in st2.items))

    print("\n4. full pipeline")
    st3 = store()
    out = remember_from_conversation(
        "We decided to freeze Moonlight as the baseline and swap in Qwen.",
        lambda m: "The team froze Moonlight as the baseline.\nThe team is trying Qwen as a swap.",
        st3)
    check("candidates extracted and added", len(out["added"]) == 2 and len(out["candidates"]) == 2)
    check("memories are retrievable afterwards",
          any("Moonlight" in h["text"] for h in st3.search("baseline model", k=5)))

    print("\n" + "=" * 72)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL EXTRACTION TESTS PASS — proposes, filters secrets/dupes, respects approval, stores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
