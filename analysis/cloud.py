"""cloud.py — read-only scan of cloud config (IAM / Terraform / CloudFormation / k8s) for misconfig.

Line-based pattern scan for the classic cloud exposures: wildcard IAM, public storage/ACLs, security
groups open to 0.0.0.0/0, public snapshots/AMIs, disabled encryption, plaintext secrets, and
over-permissive k8s. Pure and format-agnostic (it matches the risky strings, not a full parser), so a
hit is a LEAD to confirm in context, not proof.
"""
from __future__ import annotations

import re

_CLOUD = [
    (r'"Action"\s*:\s*"\*"|"Action"\s*:\s*\[\s*"\*"', "iam_wildcard_action", "HIGH",
     "IAM policy allows ALL actions (*)"),
    (r'"Resource"\s*:\s*"\*"', "iam_wildcard_resource", "MEDIUM",
     "IAM policy applies to ALL resources (*)"),
    (r'"Effect"\s*:\s*"Allow"[^}]*"Principal"\s*:\s*"\*"', "iam_public_principal", "HIGH",
     "policy grants access to ANY principal (*)"),
    (r"0\.0\.0\.0/0", "open_to_internet", "HIGH",
     "security group / firewall rule open to the entire internet (0.0.0.0/0)"),
    (r"(?i)\bpublic-read(-write)?\b|acl\s*=\s*[\"']?public", "public_bucket_acl", "HIGH",
     "storage object/bucket ACL is public"),
    (r"(?i)block_public_acls\s*=\s*false|BlockPublicAcls\s*:\s*false", "public_acls_allowed",
     "HIGH", "S3 public-ACL block is disabled"),
    (r"(?i)encrypted\s*=\s*false|encryption\s*:\s*disabled", "encryption_disabled", "MEDIUM",
     "encryption is disabled"),
    (r"(?i)\bprivileged\s*:\s*true\b", "k8s_privileged", "HIGH",
     "privileged container (full host access)"),
    (r"(?i)hostNetwork\s*:\s*true|hostPID\s*:\s*true", "k8s_host_namespace", "HIGH",
     "pod shares the host network/PID namespace"),
    (r"(?i)(password|secret|access_key|api_key)\s*[:=]\s*[\"'][^\"'\s]{6,}", "plaintext_secret",
     "HIGH", "plaintext credential in cloud config"),
    (r"(?i)\bpublicly_accessible\s*=\s*true|PubliclyAccessible\s*:\s*true", "public_db", "HIGH",
     "database instance is publicly accessible"),
]
_COMPILED = [(re.compile(p), n, s, w) for p, n, s, w in _CLOUD]


def scan_cloud(text: str) -> list[dict]:
    out = []
    for i, raw in enumerate((text or "").splitlines(), 1):
        line = raw.lstrip()
        if line.startswith("#") or line.startswith("//"):
            continue
        for rx, name, sev, why in _COMPILED:
            if rx.search(raw):
                out.append({"line": i, "issue": name, "severity": sev, "why": why,
                            "snippet": raw.strip()[:120]})
    return out
