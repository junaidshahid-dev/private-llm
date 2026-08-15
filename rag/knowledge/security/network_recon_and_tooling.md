# Network reconnaissance, tooling, and reading tool output

> source: curated-reference | version: v1 | updated: 2026-08-14 | authored

## Nmap and interpreting scans

`nmap -sV` does service/version detection; `-Pn` skips host discovery (treat the host as up);
`-sC` runs default NSE scripts. Output like `445/tcp open microsoft-ds Samba 3.0.20` tells you a
service, a version, and therefore a *candidate* vulnerability — Samba 3.0.20 is associated with the
`usermap_script` command-injection RCE (CVE-2007-2447). But **a version banner is not proof**:
banners can be spoofed, and distributions backport security fixes while leaving the version string
unchanged (a "backport"), so an old-looking version may already be patched. Before claiming a
finding, **verify** with a safe check (a targeted, non-destructive probe or an authenticated
inspection), and distinguish *observed evidence* from *inference from the banner*. This banner-vs-
reality gap is the most common source of scanner **false positives**.

## Reading other tools' output

- **tshark / Wireshark**: packet analysis. `tshark -r capture.pcap -q -z io,phs` prints the
  protocol hierarchy; conversation and follow-stream views reconstruct sessions. Look for unusual
  destinations, beaconing intervals, cleartext credentials, and DNS anomalies.
- **Burp Suite / OWASP ZAP**: intercepting proxies for web testing — capture, modify, and replay
  HTTP. Burp Intruder (or `ffuf`/`wfuzz`) automates parameter and endpoint fuzzing.
- **hashcat / John the Ripper**: offline password cracking; hashcat is GPU-accelerated and selects
  the algorithm by numeric mode.
- **Metasploit**: exploitation framework for authorised testing — modules for scanning, exploiting,
  and post-exploitation.
- **Ghidra / gdb / radare2**: reverse engineering and debugging (see `reverse_engineering.md`).

## False-positive analysis

A finding is only real if the target is actually affected. Common false positives: a version banner
that is backported/patched; a scanner flagging POODLE (an **SSLv3** attack) on a server that only
accepts TLS 1.2/1.3 with SSLv3 disabled — not vulnerable; a signature match on a page that returns
200 for everything. Always ask: does the specific precondition hold here, and can I confirm it
safely without assuming?

## Methodology and scope

Legitimate testing runs under authorisation and scope (rules of engagement): the in-scope targets,
allowed techniques (often no DoS, no social engineering unless named), a time window, data-handling
rules, and a contact. Document what you tested, keep just enough evidence to prove each finding
(the request and response), and avoid exfiltrating real sensitive data. Scope is what separates a
pentest from an intrusion.
