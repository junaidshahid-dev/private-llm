# Cryptography and cloud security

> source: curated-reference | version: v1 | updated: 2026-08-14 | authored

## Cryptography pitfalls

- **Non-constant-time comparison.** Comparing a secret (an HMAC, a token, an API key) with `==`
  short-circuits at the first differing byte, so response time leaks how many leading bytes were
  correct — a timing side channel that lets an attacker recover the value byte by byte. Use a
  constant-time compare: `hmac.compare_digest` / `secrets.compare_digest`.
- **Weak or reused secrets.** An HMAC is only as strong as its key. A short, guessable, or leaked
  HMAC/JWT secret can be brute-forced offline (e.g. with hashcat) and then used to forge valid
  signatures. Use long random secrets and rotate them.
- **Hash choice.** MD5 and SHA-1 are broken for collision resistance (practical MD5 collisions since
  2004) — never use them where collisions matter. For password storage use a slow, salted KDF —
  bcrypt, scrypt, or Argon2id — not a fast general-purpose hash. Hashes are one-way; you cannot
  "reverse" SHA-256, only brute-force or look up known inputs.
- **Cipher modes.** ECB encrypts identical plaintext blocks to identical ciphertext blocks, so it
  leaks structure (the classic "ECB penguin"). Prefer authenticated encryption (AES-GCM, or
  encrypt-then-MAC) so tampered ciphertext is rejected before decryption — this also prevents
  padding-oracle attacks, where a server that reveals padding validity (by error or timing) leaks
  one plaintext bit per query. AES keys are 128/192/256-bit; there is no "AES-512". RSA is not used
  to bulk-encrypt data — encrypt data with a symmetric key and use RSA (with OAEP padding) only to
  wrap that key.

## Cloud security and SSRF-to-metadata

Cloud instances expose an **instance metadata service** at `169.254.169.254`. On IMDSv1 a simple
GET to `/latest/meta-data/iam/security-credentials/<role>` returns temporary IAM credentials. So an
**SSRF** in an app that fetches user-supplied URLs server-side can be pointed at the metadata
endpoint to steal cloud credentials — a critical impact.

**Filter bypasses** (why a blacklist is insufficient): a blacklist blocking the strings
`169.254.169.254` and `localhost` is trivially bypassed by alternate encodings of the same IP —
decimal (`2852039166`), octal, hex (`0xA9FE A9FE`), IPv6-mapped forms, or a DNS name you control
that resolves to the metadata IP, or an open redirect / 302 to it, or DNS rebinding. The filter is
**not sufficient**.

**Correct defences:** use IMDSv2 (session-token, hop-limited), or block the metadata IP at the
network level; for the SSRF itself, allow-list destinations, resolve the hostname and reject
private/link-local ranges *after* resolution, and disallow redirects to them. Other cloud staples:
least-privilege IAM roles, no public storage buckets by default, and logging (CloudTrail) to detect
credential misuse.
