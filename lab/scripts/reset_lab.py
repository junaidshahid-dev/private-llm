"""reset_lab.py — restore the lab target to a known-good state.

    python lab/scripts/reset_lab.py            # restart the web target (fast, ephemeral reset)
    python lab/scripts/reset_lab.py --recreate # tear down and recreate from the image (full reset)

DVWA holds state in-container, so restarting the container returns it to the image's initial state;
after a restart, re-run its DB setup at http://127.0.0.1:8080/setup.php (Create / Reset Database).
This wraps docker compose so a test run always starts from the same baseline. It never touches
anything outside the lab compose project.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPOSE = os.path.join(HERE, "lab", "docker-compose.yml")


def _compose(*args) -> int:
    if not shutil.which("docker"):
        print("docker is not installed / not on PATH — cannot reset the lab.")
        return 2
    cmd = ["docker", "compose", "-f", COMPOSE, *args]
    print("+ " + " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except OSError as e:
        print(f"failed to run docker compose: {e}")
        return 2


def main() -> int:
    if not os.path.isfile(COMPOSE):
        print(f"no compose file at {COMPOSE}")
        return 2
    if "--recreate" in sys.argv:
        _compose("down", "-v")
        rc = _compose("up", "-d", "--force-recreate")
    else:
        rc = _compose("restart", "web-target")
    if rc == 0:
        print("\nlab target reset. Re-initialise DVWA at http://127.0.0.1:8080/setup.php "
              "(Create / Reset Database), then run verify_lab.py before testing.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
