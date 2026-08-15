# Secure coding

> source: curated-reference | version: v1 | updated: 2026-08-14 | authored

## Injection: never build interpreters' input by string concatenation

- **Command injection**: `os.system("ping -c 1 " + host)` lets `host = "x; rm -rf /"` run arbitrary
  shell commands. Fix by passing an argument list with no shell:
  `subprocess.run(["ping", "-c", "1", host], shell=False)`. Avoid `os.system` and `shell=True` with
  interpolation.
- **SQL injection**: use parameterised queries, never string-built SQL (see
  `web_application_security.md`).
- **Path traversal**: `open(BASE_DIR + user_path)` with `user_path = "../../etc/passwd"` escapes the
  intended directory. Resolve the final path and confirm it stays inside the allowed base
  (`os.path.commonpath`), or use the basename; reject `..` and absolute paths.

## Secrets, authentication, sessions

- **Secrets** never belong in code or git. Read them from environment variables or a secrets
  manager; add `.env` to `.gitignore`. If a secret was ever committed, **rotate it** — git history
  keeps it after deletion, and public commits are scraped within minutes.
- **Password reset / tokens**: generate with a CSPRNG (`secrets.token_urlsafe(32)`), store only a
  hash, give a short expiry, make single-use, and compare in constant time. A token that is a
  predictable function of the username (e.g. `md5(username)`) is forgeable — the vulnerability is
  predictability, not the hash per se.
- **Session cookies**: set `HttpOnly` (blocks JS theft), `Secure` (HTTPS only), and
  `SameSite=Lax`/`Strict` (limits CSRF); rotate the session id on login to prevent fixation.

## Input handling and honesty about tools/data

- Validate and canonicalise input at the boundary; prefer allow-lists over deny-lists.
- Escape/encode for the **output** context (HTML, SQL, shell, LDAP) — a single value can be safe in
  one context and dangerous in another.
- Constant-time comparison for anything secret (`hmac.compare_digest`).
- Fail closed: on error, deny rather than allow.

## HTTPS and identity misconceptions

HTTPS encrypts the URL path and query in transit, but they still land in server logs, browser
history, and `Referer` headers — so **never put secrets in a URL**. HTTPS/TLS ≥ 1.2 with modern
ciphers; a **self-signed certificate** proves possession of a key, not identity — only a CA the
client trusts vouches for identity. UTF-8 is variable width (1–4 bytes), not fixed. These are common
premises to challenge in a security discussion.
