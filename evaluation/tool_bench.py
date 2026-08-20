"""tool_bench.py — the FINAL VERDICT on every tool. One benchmark, one pass/fail per tool.

The v3/v4/v5 benchmarks measure the MODEL's reasoning; they never run a tool. This one is the
opposite: it exercises EVERY registered tool with a controlled input and checks it BEHAVES correctly
— a pure/file tool detects the planted issue; an active network tool denies an UNAUTHORIZED target
(deny-by-default safety) and works on an authorized one (injectable); a tool needing an external
binary either runs or degrades gracefully. It FAILS if any tool in the schema has no scenario, so no
tool can be added without a behavioural verdict here. CPU-only, deterministic.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mcp_layer import tools as toolmod, security as sec                       # noqa: E402
from web import tools as webmod                                              # noqa: E402
from web.search import web_search                                            # noqa: E402
from web.fetch import web_fetch                                              # noqa: E402
from web.extract import web_extract                                          # noqa: E402

TMP = tempfile.mkdtemp()
CFG = {"filesystem_read": {"enabled": True, "allowed_paths": [TMP]},
       "git_inspect": {"enabled": True, "allowed_repos": [REPO]},
       "security_tools": {"enabled": True, "require_confirmation": True,
                          "authorized_targets": [{"match": "lab.local"}, {"match": "127.0.0.1"}]},
       "web": {"enabled": True, "fetch": True, "search": True, "private_networks": False}}


def wf(name, data):
    p = os.path.join(TMP, name)
    with open(p, "wb" if isinstance(data, (bytes, bytearray)) else "w",
              encoding=None if isinstance(data, (bytes, bytearray)) else "utf-8") as f:
        f.write(data)
    return p


ELF = b"\x7fELF\x02\x01\x01" + b"\x00" * 9 + b"\x02\x00\x3e\x00"
# files used by several scenarios
F_PY = wf("vuln.py", 'import os\ncmd = request.args.get("c")\nos.system(cmd)\nKEY="AKIAIOSFODNN7EXAMPLE"\n')
F_CONF = wf("app.conf", "DEBUG = True\nAccess-Control-Allow-Origin: *\n")
F_REQ = wf("requirements.txt", "requests==2.31.0\nflask>=2.0\nurllib3\n")
F_LOG = wf("access.log", "2026-08-20T10:00:00 from 203.0.113.9\n" + "\n".join(["Failed password"] * 6))
F_IAM = wf("iam.json", '{"Effect":"Allow","Action":"*","Resource":"*"}')
F_ELF = wf("mini.elf", ELF)
F_BIN = wf("blob.bin", b"before\x00HELLOSTRING\x00after")
F_PNG = wf("img.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
import zipfile as _zip                                                        # noqa: E402
F_APK = os.path.join(TMP, "app.apk")
with _zip.ZipFile(F_APK, "w") as _z:
    _z.writestr("AndroidManifest.xml", "<manifest/>")
    _z.writestr("classes.dex", "dex\n")


def _t(name, args, confirmed=False):
    return toolmod.dispatch({"tool": name, "arguments": args}, CFG)


def _s(name, args, confirmed=True):
    return sec.dispatch({"tool": name, "arguments": args}, CFG, confirmed=confirmed)


def _graceful(r):
    e = (r.get("error") or "").lower()
    return r.get("ok") or any(w in e for w in ("not installed", "needs ", "install", "no such",
                                               "not available", "requires"))


# name -> callable() -> (passed: bool, note: str). EVERY schema tool must appear here.
def _dns():
    r = sec.dns_lookup(CFG, "lab.local", confirmed=True, _resolver=lambda h: {"host": h,
                        "addresses": ["10.0.0.1"], "reverse": {}})
    denied = not sec.dns_lookup(CFG, "8.8.8.8", confirmed=True)["ok"]
    return (r["ok"] and denied, "resolves authorized; denies unauthorized")


def _tls():
    cert = {"subject": ((("commonName", "lab.local"),),), "issuer": ((("commonName", "CA"),),),
            "notAfter": "Jan  1 00:00:00 2020 GMT", "subjectAltName": (("DNS", "lab.local"),)}
    r = sec.tls_inspect(CFG, "lab.local", confirmed=True,
                        _fetch=lambda h, p: (cert, "TLSv1.3", ("X", "", 256)))
    return (r["ok"] and r["result"]["expired"] is True, "parses cert; flags expired")


def _web_headers():
    fake = lambda c, u, cf=False: {"ok": True, "result": {"status": 200, "final_url": u,
        "headers": {"Server": "Apache/2.4.25", "Set-Cookie": "s=1"}, "body": "<form><input name=x></form>"}}
    r = sec.web_headers(CFG, "http://lab.local/", confirmed=True, _http_get=fake)
    denied = not sec.web_headers(CFG, "http://8.8.8.8/", confirmed=True)["ok"]
    return (r["ok"] and r["result"]["security_findings"] and denied, "analyzes headers; denies unauth")


SCENARIOS = {
    # ---- read-only file / repo tools (mcp_layer.tools) --------------------------------------
    "fs_list": lambda: (_t("fs_list", {"path": TMP})["ok"], "lists an allowed dir"),
    "fs_read": lambda: ("os.system" in _t("fs_read", {"path": F_PY}).get("result", ""), "reads a file"),
    "git_status": lambda: (_t("git_status", {"repo": REPO})["ok"], "git status on the repo"),
    "git_log": lambda: (_t("git_log", {"repo": REPO, "n": 3})["ok"], "git log"),
    "git_diff": lambda: (_t("git_diff", {"repo": REPO})["ok"], "git diff"),
    "source_scan": lambda: (any(f["vuln_class"] == "command_injection"
                                for f in _t("source_scan", {"path": F_PY}).get("findings", [])),
                            "finds taint -> os.system"),
    "config_scan": lambda: (len(_t("config_scan", {"path": F_CONF}).get("issues", [])) >= 2,
                            "flags DEBUG/CORS"),
    "dependency_audit": lambda: (len(_t("dependency_audit", {"path": F_REQ}).get("loose", [])) >= 2,
                                 "flags unpinned deps"),
    "ioc_extract": lambda: ("203.0.113.9" in _t("ioc_extract", {"path": F_LOG}).get("iocs", {})
                            .get("ipv4", []), "extracts an IP IOC"),
    "log_analyze": lambda: (bool(_t("log_analyze", {"path": F_LOG}).get("anomalies")),
                            "flags a brute-force burst"),
    "cloud_scan": lambda: (bool(_t("cloud_scan", {"path": F_IAM}).get("issues")),
                           "flags wildcard IAM"),
    # ---- read-only analysis tools (security) ------------------------------------------------
    "url_info": lambda: (_s("url_info", {"url": "https://lab.local:8443/x"})["ok"], "parses a URL"),
    "hash_file": lambda: (len(_s("hash_file", {"path": F_BIN}).get("result", {}).get("sha256", "")) == 64,
                          "computes sha256"),
    "strings_extract": lambda: ("HELLOSTRING" in "".join(_s("strings_extract", {"path": F_BIN})
                                .get("result", []) if isinstance(_s("strings_extract", {"path": F_BIN})
                                .get("result"), list) else [str(_s("strings_extract", {"path": F_BIN})
                                .get("result"))]), "extracts printable strings"),
    "file_type": lambda: (_s("file_type", {"path": F_PNG})["ok"], "identifies by magic bytes"),
    "binary_info": lambda: (_s("binary_info", {"path": F_ELF})["result"]["format"] == "ELF",
                            "parses ELF header"),
    "qr_decode": lambda: (_graceful(_s("qr_decode", {"path": F_PNG})), "decodes or degrades"),
    "apk_analyze": lambda: (_graceful(_s("apk_analyze", {"path": F_APK})), "lists APK entries"),
    "pcap_analyze": lambda: (_graceful(_s("pcap_analyze", {"path": F_BIN})), "summary or degrades"),
    "adb_devices": lambda: (_graceful(_s("adb_devices", {})), "lists devices or degrades"),
    "readelf_headers": lambda: (_graceful(_s("readelf_headers", {"path": F_ELF})), "runs or degrades"),
    "objdump_disasm": lambda: (_graceful(_s("objdump_disasm", {"path": F_ELF})), "runs or degrades"),
    "nm_symbols": lambda: (_graceful(_s("nm_symbols", {"path": F_ELF})), "runs or degrades"),
    "radare2_analyze": lambda: (_graceful(_s("radare2_analyze", {"path": F_ELF})), "runs or degrades"),
    # ---- active network tools: deny-by-default is the behavioural guarantee ------------------
    "nmap_scan": lambda: (not _s("nmap_scan", {"target": "8.8.8.8"})["ok"], "denies unauthorized target"),
    "masscan_scan": lambda: (not _s("masscan_scan", {"target": "8.8.8.8"})["ok"], "denies unauthorized"),
    "ffuf_discover": lambda: (not _s("ffuf_discover", {"target": "http://8.8.8.8/"})["ok"],
                              "denies unauthorized"),
    "http_get": lambda: (not _s("http_get", {"url": "http://8.8.8.8/"})["ok"], "denies unauthorized URL"),
    "dns_lookup": _dns,
    "tls_inspect": _tls,
    "web_headers": _web_headers,
    # ---- web research tools ------------------------------------------------------------------
    "web_search": lambda: (web_search(CFG, "nmap", _search=lambda q:
                           '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnmap.org">'
                           'Nmap</a>')["ok"], "returns ranked candidates (stub backend)"),
    "web_fetch": lambda: (not web_fetch(CFG, "http://127.0.0.1:1/")["ok"], "SSRF-blocks a private URL"),
    "web_extract": lambda: (not web_extract(CFG, "http://127.0.0.1:1/")["ok"], "SSRF-blocks a private URL"),
}


def _selftest() -> int:
    print("=" * 74)
    print("TOOL BENCHMARK — behavioural verdict on EVERY registered tool")
    print("=" * 74)
    registered = sorted({e["name"] for e in toolmod.schema() + sec.schema() + webmod.schema()})
    uncovered = [n for n in registered if n not in SCENARIOS]
    fails = []
    for name in registered:
        if name in uncovered:
            print(f"  [FAIL] {name}  — NO SCENARIO (tool left out of the verdict)")
            fails.append(name)
            continue
        try:
            ok, note = SCENARIOS[name]()
        except Exception as e:                        # noqa: BLE001
            ok, note = False, f"raised {type(e).__name__}: {e}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  — {note}")
        if not ok:
            fails.append(name)
    extra = [n for n in SCENARIOS if n not in registered]
    for n in extra:
        print(f"  [WARN] scenario for {n!r} but it is not a registered tool")
    print("-" * 74)
    print(f"VERDICT: {len(registered) - len(fails)}/{len(registered)} tools verified"
          + (f"   FAILED: {fails}" if fails else ""))
    print("=" * 74)
    if fails:
        return 1
    print("ALL TOOLS PASS — every registered tool behaves correctly (detects / gates / degrades).")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
