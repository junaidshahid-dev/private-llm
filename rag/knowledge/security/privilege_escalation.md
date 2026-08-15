# Privilege escalation

> source: curated-reference | version: v1 | updated: 2026-08-14 | authored

## Linux

Enumeration first: kernel/OS version, `sudo -l`, SUID/SGID binaries
(`find / -perm -4000 2>/dev/null`), writable files owned by root, cron jobs, capabilities
(`getcap -r /`), running services, and PATH/`LD_PRELOAD` weaknesses.

**Sudo misconfigurations.** If `sudo -l` shows a binary runnable as root with `NOPASSWD`, many
binaries can spawn a shell or read/write arbitrary files as root — this is what **GTFOBins**
catalogues. Examples: a pager like `less` or `more` run as root can execute `!/bin/sh` from its
interactive prompt to spawn a root shell; `find ... -exec /bin/sh \;`, `vim`/`nano` shell escapes,
`awk 'BEGIN{system("/bin/sh")}'`. So "user may run /usr/bin/less as root, NOPASSWD" is exploitable:
open it as root and invoke a shell from the pager. Fix by removing the sudo entry or restricting to
a specific, non-interactive command with no shell escape.

**SUID binaries.** A SUID-root binary runs with root privileges. If it is a GTFOBins-listed program
(or calls one, or trusts a relative PATH), it can be leveraged to run code as root. Custom SUID
binaries that call `system()` with a relative path are exploitable via PATH hijacking.

**Other vectors.** Writable cron scripts run by root; writable `/etc/passwd` or a service unit;
kernel exploits (last resort, risky); credentials in config files, history, or environment.

## Windows

Enumeration: `whoami /priv` (token privileges like `SeImpersonatePrivilege`), unquoted service
paths, weak service permissions (`sc qc`, ACLs), AlwaysInstallElevated, stored credentials,
scheduled tasks, and unpatched kernel/service CVEs. `SeImpersonatePrivilege` enables the "potato"
family of local escalations. Active Directory escalation and lateral movement are covered in
`active_directory.md`.

## Principle

Privilege escalation almost always comes from a **misconfiguration or over-permission**, not a
mysterious exploit. The fix is least privilege: remove the unneeded SUID bit, tighten the sudo
rule, quote the service path, correct the ACL.
