"""killswitch.py — the OPERATOR's control for the global STOP (never a model tool).

    python scripts/killswitch.py engage "reason"     # halt everything now
    python scripts/killswitch.py status              # is it engaged?
    python scripts/killswitch.py clear               # lift the halt (operator action)

Only a human runs this. The model has no path to engage or clear the switch.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_layer import killswitch                                             # noqa: E402


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "engage":
        reason = " ".join(sys.argv[2:]) or "operator stop"
        print(killswitch.engage(reason, by="operator-cli"))
        print("ALL OPERATIONS HALTED. Run 'clear' to resume.")
    elif cmd == "clear":
        # this invocation IS the human operator's explicit acknowledgement
        print(killswitch.clear(operator_ack=True))
    elif cmd == "status":
        print(killswitch.status())
    else:
        print(f"unknown command {cmd!r}; use engage | status | clear")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
