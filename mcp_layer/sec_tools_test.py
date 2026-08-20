"""sec_tools_test.py — the new security tools: real read-only analysis + gated active recon.

    python mcp_layer/sec_tools_test.py

The read-only file tools (hash/strings/file_type) need no external binary, so they are tested
end-to-end. The active tools (masscan/ffuf) and adb need binaries, so their SAFETY paths are tested
instead: an unauthorized target is denied, an authorized-but-unconfirmed run is blocked, and a
missing binary is reported cleanly — the gates that matter, provable without the binary.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mcp_layer import security as sec                                         # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("SECURITY TOOLS TEST — read-only analysis works; active tools stay gated")
    print("=" * 74)

    tmp = tempfile.mkdtemp()
    cfg = {"filesystem_read": {"enabled": True, "allowed_paths": [tmp]},
           "security_tools": {"enabled": True, "require_confirmation": True,
                              "authorized_targets": [{"id": "lab", "match": "10.10.10.5",
                                                      "note": "my lab host"}]}}
    disabled = {"security_tools": {"enabled": False}}

    def wf(name, data: bytes):
        p = os.path.join(tmp, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    # ---- hash_file (real, read-only) ----------------------------------------
    print("\n1. hash_file — real hashes, path-confined, gated")
    payload = b"malware sample bytes"
    hp = wf("sample.bin", payload)
    r = sec.hash_file(cfg, hp)
    check("sha256 matches hashlib", r["ok"] and
          r["result"]["sha256"] == hashlib.sha256(payload).hexdigest())
    check("also returns md5 and sha1", "md5" in r["result"] and "sha1" in r["result"])
    check("path outside allowed roots is denied",
          not sec.hash_file(cfg, os.path.join(HERE, "MODEL_SPEC.lock.json"))["ok"])
    check("disabled group denies", not sec.hash_file(disabled, hp)["ok"])

    # ---- strings_extract (real) ---------------------------------------------
    print("\n2. strings_extract — printable runs, min length honoured")
    sp = wf("s.bin", b"AB\x00HELLOworld\x01\x02config123\x00xx")
    r = sec.strings_extract(cfg, sp, min_len=4)
    strings = r["result"]["strings"]
    check("extracts a long run", "HELLOworld" in strings)
    check("extracts a second run", "config123" in strings)
    check("drops sub-minimum runs (2-char 'AB')", "AB" not in strings)

    # ---- file_type (real magic bytes) ---------------------------------------
    print("\n3. file_type — magic-byte identification")
    for name, magic, want in [("e", b"\x7fELF\x02\x01\x01", "ELF"), ("p", b"MZ\x90\x00", "PE"),
                              ("d", b"%PDF-1.7", "PDF"), ("z", b"PK\x03\x04\x14", "ZIP")]:
        r = sec.file_type(cfg, wf(name, magic))
        check(f"{want} identified", want in r["result"]["type"], r["result"]["type"])

    # ---- masscan / ffuf: the gates (no binary needed to prove these) --------
    print("\n4. active recon — authorization + confirmation gates")
    unauth = sec.masscan_scan(cfg, "1.2.3.4", confirmed=True)
    check("unauthorized target denied", not unauth["ok"] and "not authorized" in unauth["error"])
    unconf = sec.masscan_scan(cfg, "10.10.10.5", confirmed=False)
    check("authorized but unconfirmed => needs_confirmation", unconf.get("needs_confirmation"))
    confd = sec.masscan_scan(cfg, "10.10.10.5", confirmed=True)
    check("authorized + confirmed proceeds to the tool (installed? report cleanly)",
          confd["ok"] or "not installed" in confd.get("error", ""), str(confd.get("error")))
    ff = sec.ffuf_discover(cfg, "10.10.10.5", confirmed=False)
    check("ffuf also gated by confirmation", ff.get("needs_confirmation"))
    check("ffuf denies an unauthorized URL",
          not sec.ffuf_discover(cfg, "http://evil.example", confirmed=True)["ok"])

    # ---- adb (read-only local) ----------------------------------------------
    print("\n5. adb_devices — read-only, gated by the group")
    check("disabled group denies adb", not sec.adb_devices(disabled)["ok"])
    r = sec.adb_devices(cfg)
    check("enabled: returns devices or a clean 'not installed'",
          r["ok"] or "not installed" in r.get("error", ""), str(r.get("error")))

    # ---- dispatch routing ----------------------------------------------------
    print("\n5b. binary_info (pure header parse) + gated RE tools")
    elf = wf("mini.elf", b"\x7fELF\x02\x01\x01" + b"\x00" * 9 + b"\x02\x00\x3e\x00")
    bi = sec.binary_info(cfg, elf)
    check("binary_info identifies ELF x86-64", bi["ok"] and bi["result"]["format"] == "ELF"
          and bi["result"]["machine"] == "x86-64", str(bi.get("result")))
    check("binary_info is path-confined",
          not sec.binary_info(cfg, os.path.join(HERE, "MODEL_SPEC.lock.json"))["ok"])
    re_r = sec.readelf_headers(cfg, elf)
    check("readelf_headers runs OR degrades gracefully when absent",
          re_r["ok"] or "not installed" in re_r.get("error", ""), str(re_r)[:80])
    check("binary_info denied when the group is disabled", not sec.binary_info(disabled, elf)["ok"])

    print("\n5c. dns_lookup + tls_inspect (gated recon, injectable for testing)")
    dl = sec.dns_lookup(cfg, "10.10.10.5", confirmed=True,
                        _resolver=lambda h: {"host": h, "addresses": ["10.10.10.5"], "reverse": {}})
    check("dns_lookup returns records for an authorized target", dl["ok"]
          and dl["result"]["addresses"] == ["10.10.10.5"])
    check("dns_lookup needs confirmation", sec.dns_lookup(cfg, "10.10.10.5")["ok"] is False)
    check("dns_lookup denies an unauthorized target",
          sec.dns_lookup(cfg, "8.8.8.8", confirmed=True)["ok"] is False)
    fake_cert = {"subject": ((("commonName", "lab.local"),),),
                 "issuer": ((("commonName", "Lab CA"),),),
                 "notAfter": "Jan  1 00:00:00 2020 GMT",
                 "subjectAltName": (("DNS", "lab.local"),)}
    ti = sec.tls_inspect(cfg, "10.10.10.5", confirmed=True,
                         _fetch=lambda h, p: (fake_cert, "TLSv1.3", ("TLS_AES_256_GCM_SHA384", "", 256)))
    check("tls_inspect summarizes the certificate", ti["ok"]
          and ti["result"]["subject_cn"] == "lab.local" and ti["result"]["tls_version"] == "TLSv1.3")
    check("tls_inspect flags an expired cert", ti["result"]["expired"] is True)

    print("\n5d. web_headers (fetch via gated http_get, then analyze)")
    fake_get = lambda c, u, cf=False: {"ok": True, "result": {
        "status": 200, "final_url": "http://web-target/", "content_type": "text/html",
        "headers": {"Server": "Apache/2.4.25", "Set-Cookie": "sid=abc"},
        "body": '<form action="/login" method="post"><input name="user"></form>'}}
    wh = sec.web_headers(cfg, "http://10.10.10.5/", confirmed=True, _http_get=fake_get)
    check("web_headers reports security findings + attack surface",
          wh["ok"] and wh["result"]["security_findings"] and wh["result"]["attack_surface"]["forms"])
    check("web_headers surfaces missing headers + server disclosure",
          any(f["issue"] == "server_version_disclosure" for f in wh["result"]["security_findings"]))
    check("web_headers denies an unauthorized URL (via the http_get gate)",
          not sec.web_headers(cfg, "http://8.8.8.8/", confirmed=True)["ok"])

    print("\n6. dispatch routes the new tools")
    check("dispatch runs hash_file",
          sec.dispatch({"tool": "hash_file", "arguments": {"path": hp}}, cfg)["ok"])
    check("dispatch rejects an unknown tool",
          not sec.dispatch({"tool": "nope", "arguments": {}}, cfg)["ok"])
    names = {t["name"] for t in sec.schema()}
    check("schema advertises the new tools",
          {"hash_file", "strings_extract", "file_type", "masscan_scan", "ffuf_discover",
           "adb_devices"} <= names)

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL SECURITY-TOOL TESTS PASS — analysis works, active tools stay authorization-gated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
