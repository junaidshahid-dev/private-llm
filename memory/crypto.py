"""crypto.py — encryption at rest for the memory store (Phase 11.7).

Real AES via cryptography.Fernet when available (it is), with a local key file (gitignored) or a
MEMORY_KEY env var. The store's JSON is encrypted on disk so a stolen file is not readable. If the
library is somehow absent, callers fall back to plaintext with a clear signal — never a silent
false sense of security.
"""
from __future__ import annotations

import os

DEFAULT_KEYFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".memory_key")


def available() -> bool:
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False


def load_or_create_key(keyfile: str = DEFAULT_KEYFILE) -> bytes:
    """MEMORY_KEY env var wins; else a local key file (created 0600 on first use)."""
    env = os.environ.get("MEMORY_KEY")
    if env:
        return env.encode()
    if os.path.exists(keyfile):
        return open(keyfile, "rb").read().strip()
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    os.makedirs(os.path.dirname(keyfile), exist_ok=True)
    with open(keyfile, "wb") as f:
        f.write(key)
    try:
        os.chmod(keyfile, 0o600)
    except OSError:
        pass
    return key


def encrypt(data: bytes, key: bytes) -> bytes:
    from cryptography.fernet import Fernet
    return Fernet(key).encrypt(data)


def decrypt(blob: bytes, key: bytes) -> bytes:
    from cryptography.fernet import Fernet
    return Fernet(key).decrypt(blob)
