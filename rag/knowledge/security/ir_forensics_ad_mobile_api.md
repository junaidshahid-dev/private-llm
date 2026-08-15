# Incident response, forensics, Active Directory, mobile, API, detection

> source: curated-reference | version: v1 | updated: 2026-08-14 | authored

## Incident response order (and why order matters)

1. **Preserve volatile evidence first.** Capture memory and current state before anything changes
   it — running processes, network connections, logged-in users, and a memory image. Do **not**
   reboot or power off first: RAM (and thus running malware, keys, and connections) is lost on
   reboot. Rebooting or wiping immediately destroys the evidence that tells you what happened.
2. **Scope from logs.** Build a timeline from web/access/auth logs and endpoint telemetry; find
   what else was touched and whether there was lateral movement.
3. **Contain.** Isolate at the network level (block/segment) rather than wiping — keep the system
   for analysis while stopping the bleeding.
4. **Eradicate and recover.** Rebuild from a known-good state rather than only deleting artifacts
   (you rarely see everything the attacker did), and **rotate all credentials** the compromised
   host could reach.

For a web shell found days ago: preserve the file and logs first, scope the entry point and
subsequent actions from logs, contain, then rebuild and rotate. Assume the attacker had the web
server's privileges.

## Digital forensics and detection engineering

- **Beaconing / C2**: a host making small, regular outbound connections to the same unfamiliar
  destination on a fixed interval is classic command-and-control beaconing. Investigate the owning
  process, its binary path/hash and parent, and persistence — don't power off (lose memory).
- **Password spraying**: one source trying a few passwords across *many* accounts (each username
  once) to evade lockouts. The detecting metric is high **distinct-username count per source** with
  clustered failures — not per-account failure counts. Rate-limit by source and require MFA.
- Distinguish **observed evidence** (a log line, a captured packet, a file on disk) from
  **inference** (what it probably means). Report findings with the evidence.

## Active Directory

- **Kerberoasting**: request a Kerberos TGS ticket for a service account (one with an SPN); the
  ticket is encrypted with the service account's password hash, so it can be cracked **offline** to
  recover the password. Mitigate with long/managed service-account passwords and monitoring.
- **AS-REP roasting**: accounts with pre-authentication disabled can have crackable material
  requested without credentials.
- Common post-compromise: Pass-the-Hash, DCSync, and abusing delegation. Tiered admin and least
  privilege limit blast radius.

## Mobile / Android

An APK is a zip: `AndroidManifest.xml`, `classes.dex`, resources, and `lib/` native code. Static
analysis reads requested **permissions** and components. A red-flag pattern: an app requesting
`SEND_SMS`, `READ_CONTACTS`, `RECEIVE_SMS`, with no launcher activity and a `BOOT_COMPLETED`
receiver (starts hidden on boot) — consistent with an SMS trojan / stealthy malware. Full manifest
and permission decoding uses tools like androguard/apktool/jadx.

## API security

- **JWT `alg:none`**: a server that accepts a token whose header sets `"alg":"none"` and honours it
  without verifying a signature lets anyone forge tokens — an authentication bypass. Also reject the
  algorithm-confusion attack (RS256 verified as HS256 using the public key as an HMAC secret).
- **BOLA/IDOR**: same object-level authorization flaw as web (see `web_application_security.md`).
- Enforce authentication and authorization on every endpoint, validate input, and rate-limit.
