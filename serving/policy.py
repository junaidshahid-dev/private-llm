"""policy.py — build the system prompt for the configurable behavioural layer.

    from serving.policy import system_prompt
    sys_msg = system_prompt()                 # uses active_mode from configs/behavior_policy.yaml
    sys_msg = system_prompt(mode="off")       # or override per call

    python serving/policy.py                   # print each mode's prompt to inspect

WHY THIS EXISTS
Behaviour is not trained into the weights — the adapter is pure capability. The decline policy
(if any) is assembled here from configs/behavior_policy.yaml and prepended as a system message at
inference. That means you change what the model will and won't do by editing a YAML file, not by
retraining. The honesty rules (no fabrication, no pretending it ran a tool) are in every mode
because they are accuracy, not policy.

This keeps the operator in control: capability_first for your own use, strict if you ever expose
it to others, off for the raw model — all without touching the weights.
"""
from __future__ import annotations

import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_PATH = os.path.join(HERE, "configs", "behavior_policy.yaml")


def load_policy():
    import yaml
    with open(POLICY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def system_prompt(mode: str | None = None) -> str:
    """Assemble the system prompt for the given mode (or the active_mode from the config)."""
    pol = load_policy()
    mode = mode or pol.get("active_mode", "capability_first")
    if mode not in pol["modes"]:
        raise ValueError(f"unknown mode {mode!r}; choose from {list(pol['modes'])}")
    mode_cfg = pol["modes"][mode]
    # A mode can opt out of the always-on rules (that is what makes "off" a truly raw model,
    # not "honesty rules only"). Everything else keeps them, since they are accuracy not policy.
    base = list(pol.get("always", [])) if mode_cfg.get("include_always", True) else []
    lines = base + list(mode_cfg.get("extra", []))
    # Collapse the wrapped YAML scalars into single clean sentences. Empty -> no system prompt.
    return "\n".join("- " + " ".join(l.split()) for l in lines)


def main() -> int:
    pol = load_policy()
    print("=" * 74)
    print(f"BEHAVIOUR POLICY — active_mode: {pol.get('active_mode')}")
    print("=" * 74)
    for mode in pol["modes"]:
        print(f"\n### mode: {mode}   {'(ACTIVE)' if mode == pol.get('active_mode') else ''}")
        print(f"  note: {' '.join(pol['modes'][mode].get('note', '').split())}")
        print("  system prompt:")
        for line in system_prompt(mode).splitlines():
            print(f"    {line}")
    print("\n" + "=" * 74)
    print("Change behaviour by editing configs/behavior_policy.yaml — no retraining.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
