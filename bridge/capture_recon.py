"""capture_recon.py — run the real security tools and SAVE their output in the loop's format.

    python bridge/capture_recon.py                     # default: nmap + ffuf against the lab
    python bridge/capture_recon.py --out mycap.json

The bridge for the co-location gap: the model needs a GPU (Kaggle) and the lab needs Docker
(local), and they aren't on the same machine. So we split the loop — capture REAL tool results
here (local, no model), then let Moonlight interpret them on Kaggle (bridge/interpret_recon.py).

This runs each tool and structures the result exactly like the MCP executor / controller.interpret
expects — a list of {"tool", "result": {"ok", "command", "output"}} — so the model reasons over the
same shape it would in the live loop. No model, no GPU; just the tools you already validated.

Default steps target the lab via `docker exec lab-operator ...` (nmap then ffuf). Commit/push the
resulting JSON, `git pull` on Kaggle, and run interpret_recon.py there.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(HERE, "bridge", "recon_capture.json")
STEP_TIMEOUT = 300

# (tool name as the model knows it, the command to actually run). Matches the live-tested lab flow.
DEFAULT_STEPS = [
    ("nmap_scan", ["docker", "exec", "lab-operator", "nmap", "-sV", "web-target"]),
    ("ffuf_discover", ["docker", "exec", "lab-operator", "ffuf", "-s",
                       "-u", "http://web-target/FUZZ", "-w", "/usr/share/wordlists/common.txt"]),
]


def run_step(tool: str, cmd: list[str]) -> dict:
    """Run one command; structure it like a real MCP tool result. Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=STEP_TIMEOUT)
        out = ((r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")).strip()
        return {"tool": tool, "result": {"ok": r.returncode == 0, "command": " ".join(cmd),
                                         "output": out[:8000]}}
    except FileNotFoundError:
        return {"tool": tool, "result": {"ok": False, "command": " ".join(cmd),
                                         "error": "command not found (is docker on PATH?)"}}
    except subprocess.TimeoutExpired:
        return {"tool": tool, "result": {"ok": False, "command": " ".join(cmd),
                                         "error": f"timed out after {STEP_TIMEOUT}s"}}


def capture(steps=DEFAULT_STEPS, target: str = "web-target (DVWA lab)") -> dict:
    results = [run_step(tool, cmd) for tool, cmd in steps]
    return {"target": target,
            "task": ("Assess the authorized lab target. Interpret the real tool results below, "
                     "correlate across them, distinguish observed evidence from inference and real "
                     "findings from false positives, and state what you would test next."),
            "results": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    cap = capture()
    for r in cap["results"]:
        res = r["result"]
        flag = "ok" if res.get("ok") else "ERROR"
        print(f"[{flag}] {r['tool']}: {res.get('command')}")
        print("       " + (res.get("output") or res.get("error") or "")[:160].replace("\n", " "))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(cap, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nsaved {len(cap['results'])} tool results -> {os.path.relpath(args.out, HERE)}")
    print("commit + push this file, then run bridge/interpret_recon.py on Kaggle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
