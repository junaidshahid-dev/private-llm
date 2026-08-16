"""features_test.py — the Phase-11 gap closers: encryption, rollback, importance, conflicts, seed.

    python memory/features_test.py
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
from memory.extract import classify_importance, detect_conflicts, process_candidates  # noqa: E402
from memory.project_seed import seed                                         # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def tmp(name="m.json"):
    return os.path.join(tempfile.mkdtemp(), name)


def main() -> int:
    print("=" * 72)
    print("PHASE-11 FEATURES TEST — encryption, rollback, importance, conflicts, project seed")
    print("=" * 72)

    print("\n1. encryption at rest")
    d = tempfile.mkdtemp()
    kf = os.path.join(d, "key")
    enc = MemoryStore(path=os.path.join(d, "e.json"), project="p", encrypt=True, keyfile=kf)
    check("encryption is active", enc.encrypt is True)
    enc.add("The base model is Moonlight-16B, a sensitive project detail.")
    raw = open(enc.path, "rb").read()
    check("file on disk is NOT plaintext (ciphertext)", b"Moonlight" not in raw and len(raw) > 0)
    enc2 = MemoryStore(path=enc.path, project="p", encrypt=True, keyfile=kf)
    check("reloads and decrypts with the key",
          any("Moonlight" in i["text"] for i in enc2.items))

    print("\n2. rollback / restore")
    s = MemoryStore(path=tmp(), project="p")
    old = s.add("The base model is Moonlight.")["id"]
    new = s.supersede(old, "The base model is Qwen.")["id"]
    rb = s.rollback(old)
    check("rollback succeeds", rb["ok"])
    check("old fact is CURRENT again after rollback",
          any(h["id"] == old for h in s.search("base model", k=5)))
    check("the replacement is archived by rollback", s.get(new)["archived"] is True)
    fid = s.add("a note to forget then restore")["id"]
    s.forget(fid)
    s.restore(fid)
    check("restore un-forgets", any(h["id"] == fid for h in s.search("note forget restore", k=5)))

    print("\n3. importance classifier")
    hi = classify_importance("We decided the architecture must always use the relevance gate.")
    loo = classify_importance("might try something later today, currently unsure")
    check("a decision scores higher than a temporary statement", hi > loo, f"{hi} vs {loo}")

    print("\n4. conflict detection (same subject, different value)")
    c = MemoryStore(path=tmp(), project="p")
    c.add("The project base model is Moonlight.")
    conf = detect_conflicts("The project base model is Qwen.", c)
    check("conflict detected", len(conf) >= 1, str([x['text'] for x in conf]))
    check("an exact duplicate is NOT flagged as a conflict",
          detect_conflicts("The project base model is Moonlight.", c) == [])
    res = process_candidates(["The project base model is Qwen."], c)
    check("conflicting candidate is SURFACED, not blindly stored",
          res["conflicts"] and "The project base model is Qwen." not in res["added"])

    print("\n5. seeded private-llm project memory (why did we stop SFT?)")
    ps = MemoryStore(path=tmp(), project="private-llm")
    r = seed(ps)
    check("seeded the project facts", r["added"] >= 8, str(r))
    sft = ps.search("why did we stop SFT fine-tuning", k=3)
    check("retrieves the actual SFT decision",
          any("SFT" in h["text"] or "degraded" in h["text"] for h in sft),
          sft[0]["text"][:60] if sft else "none")
    base = ps.search("what is the current baseline model", k=3)
    check("retrieves the baseline decision", any("baseline" in h["text"].lower() for h in base))
    check("re-seeding is idempotent (dedup)", seed(ps)["added"] == 0)

    print("\n" + "=" * 72)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL PHASE-11 FEATURE TESTS PASS — encrypted, rollbackable, classified, conflict-aware, seeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
