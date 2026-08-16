"""store_test.py — the memory core: add, secret-refusal, scored retrieval, isolation, versioning.

    python memory/store_test.py
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

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def fresh():
    return MemoryStore(path=os.path.join(tempfile.mkdtemp(), "m.json"), project="proj-a")


def main() -> int:
    print("=" * 72)
    print("MEMORY STORE TEST — typed memory, secrets refused, scored + isolated retrieval, versioning")
    print("=" * 72)

    print("\n1. add + types")
    m = fresh()
    r = m.add("The project uses Moonlight-16B as the base model.", mtype="semantic", importance=0.9)
    check("add returns an id", r["ok"] and r["id"])
    check("bad type falls back to semantic",
          m.add("x", mtype="nonsense")["ok"] and m.get(m.items[-1]["id"])["type"] == "semantic")
    check("empty memory refused", not m.add("   ")["ok"])

    print("\n2. NEVER store secrets")
    s = m.add("remember my aws key AKIAIOSFODNN7EXAMPLE for the bucket")
    check("secret is refused", not s["ok"] and "secret" in s["error"])
    check("secret did NOT get stored",
          not any("AKIAIOSFODNN7EXAMPLE" in i["text"] for i in m.items))

    print("\n3. scored retrieval (relevance + importance + recency + confidence)")
    m.add("I prefer verified answers over confident guesses.", importance=0.8)
    m.add("The lab web target is DVWA on port 8080.", importance=0.5)
    hits = m.search("what base model does the project use", k=3)
    check("relevant memory surfaces first", "Moonlight" in hits[0]["text"], hits[0]["text"][:40])
    check("carries a score", "_score" in hits[0])
    check("irrelevant query returns little", len(m.search("banana recipes", k=3)) == 0
          or all("Moonlight" not in h["text"] for h in m.search("banana recipes", k=3)))

    print("\n4. project isolation (no cross-project leakage)")
    m.add("SECRET-PROJECT-B fact about client X", project="proj-b", importance=0.9)
    a_hits = m.search("client X fact", project="proj-a", k=5)
    check("project A search does NOT see project B memory",
          all("PROJECT-B" not in h["text"] for h in a_hits))
    b_hits = m.search("client X fact", project="proj-b", k=5)
    check("project B search DOES see its own memory",
          any("PROJECT-B" in h["text"] for h in b_hits))

    print("\n5. conflict resolution — supersede (current vs history)")
    old = m.add("The project uses Moonlight.", mtype="semantic")["id"]
    sup = m.supersede(old, "The project switched to Qwen2.5-Coder-14B.")
    check("supersede returns the new id + the superseded one", sup["ok"] and sup["superseded"] == old)
    cur = m.search("what model does the project use now", k=5)
    check("CURRENT shows the new fact", any("Qwen" in h["text"] for h in cur))
    check("superseded old fact is NOT returned as current",
          all(h["id"] != old for h in cur))
    check("but the old fact is kept in HISTORY", any(h["id"] == old for h in m.history()))

    print("\n6. forget + expiry")
    fid = m.add("temporary note to forget", importance=0.4)["id"]
    m.forget(fid)
    check("forgotten memory drops out of retrieval",
          all(h["id"] != fid for h in m.search("temporary note", k=5)))
    m.add("expired fact", expires=1.0)   # epoch 1970 -> already expired
    check("expired memory not returned", not any("expired fact" in h["text"]
          for h in m.search("expired fact", k=5)))

    print("\n7. poison-safe context block")
    blk = m.context_block(m.search("model", k=2))
    check("context labels memories as DATA not instructions",
          "not instructions" in blk and "REMEMBERED" in blk)

    print("\n8. persistence")
    path = m.path
    m2 = MemoryStore(path=path, project="proj-a")
    check("reloads from disk", any("Moonlight" in i["text"] or "Qwen" in i["text"]
          for i in m2.items))

    print("\n" + "=" * 72)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL MEMORY-STORE TESTS PASS — typed, secret-safe, isolated, versioned, poison-safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
