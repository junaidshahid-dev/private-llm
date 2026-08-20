"""cloud_test.py — cloud-config misconfiguration detection.

    python analysis/cloud_test.py
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from analysis.cloud import scan_cloud                                        # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 70)
    print("CLOUD CONFIG SCAN — IAM / storage / network / secrets misconfig")
    print("=" * 70)

    iam = scan_cloud('{"Effect":"Allow","Action":"*","Resource":"*"}')
    issues = {x["issue"] for x in iam}
    check("flags wildcard IAM action", "iam_wildcard_action" in issues, str(issues))
    check("flags wildcard IAM resource", "iam_wildcard_resource" in issues)

    net = scan_cloud("cidr_blocks = [\"0.0.0.0/0\"]")
    check("flags 0.0.0.0/0 open rule", any(x["issue"] == "open_to_internet" for x in net))

    s3 = scan_cloud("acl = \"public-read\"\nblock_public_acls = false")
    s3i = {x["issue"] for x in s3}
    check("flags public bucket ACL", "public_bucket_acl" in s3i)
    check("flags disabled public-ACL block", "public_acls_allowed" in s3i)

    k8s = scan_cloud("securityContext:\n  privileged: true\nhostNetwork: true")
    k8si = {x["issue"] for x in k8s}
    check("flags privileged k8s container", "k8s_privileged" in k8si)
    check("flags host namespace sharing", "k8s_host_namespace" in k8si)

    sec = scan_cloud('password = "SuperSecretValue123"')
    check("flags a plaintext secret", any(x["issue"] == "plaintext_secret" for x in sec))

    check("a commented risky line does NOT flag",
          scan_cloud('# Action: "*" in a comment') == [])
    check("a safe config is clean",
          scan_cloud('encrypted = true\ncidr_blocks = ["10.0.0.0/8"]') == [])

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL CLOUD-SCAN TESTS PASS — flags IAM/storage/network/k8s/secret misconfig, spares safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
