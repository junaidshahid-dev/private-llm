"""safety.py — the web layer's security boundary: SSRF-safe URL validation + a capped raw fetch.

web_fetch is for RESEARCH — arbitrary PUBLIC URLs — which is the opposite trust model from the
security tools' authorized-target allowlist. Here the danger is SSRF: a URL (or a redirect, or a
DNS-rebind) pointing at YOUR private network, localhost, or the cloud metadata endpoint. So the rule
is default-DENY of non-public addresses:

    scheme must be http/https
      -> resolve the host to an IP
      -> the IP must be PUBLIC (not private / loopback / link-local / reserved / multicast)
      -> re-check on EVERY redirect hop (DNS-rebinding / open-redirect defense)

169.254.169.254 (cloud metadata) is link-local, so it is blocked by this rule automatically.

Pure stdlib. resolver and fetch are injectable so the SSRF logic is unit-tested without a network.
"""
from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

TIMEOUT = 15
MAX_BODY = 2_000_000
MAX_REDIRECTS = 5
UA = "private-llm/web_fetch (research; treats page content as untrusted data)"


def resolve_ip(host: str, resolver=None):
    resolver = resolver or socket.gethostbyname
    try:
        return resolver(host)
    except (OSError, UnicodeError, ValueError):
        return None


def is_public_ip(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
                or a.is_multicast or a.is_unspecified)


def validate_public_url(url: str, resolver=None):
    """(ok, host, ip, reason). ok only when the URL is http/https AND resolves to a PUBLIC IP."""
    try:
        p = urlparse((url or "").strip())
    except ValueError:
        return False, None, None, "unparseable URL"
    if p.scheme not in ("http", "https"):
        return False, None, None, f"scheme {p.scheme or '(none)'!r} not allowed (http/https only)"
    host = p.hostname
    if not host:
        return False, None, None, "no host in URL"
    ip = resolve_ip(host, resolver)
    if ip is None:
        return False, host, None, f"cannot resolve host {host!r}"
    if not is_public_ip(ip):
        return False, host, ip, f"host {host} resolves to non-public IP {ip} — blocked (SSRF/rebind)"
    return True, host, ip, ""


def check_url(url: str, allow_private: bool = False, resolver=None):
    """(ok, reason). Full SSRF validation unless the operator explicitly allowed private networks."""
    if allow_private:
        try:
            p = urlparse((url or "").strip())
        except ValueError:
            return False, "unparseable URL"
        if p.scheme not in ("http", "https"):
            return False, f"scheme {p.scheme or '(none)'!r} not allowed"
        return (True, "") if p.hostname else (False, "no host in URL")
    ok, _host, _ip, reason = validate_public_url(url, resolver)
    return ok, reason


def safe_fetch(url: str, allow_private: bool = False, resolver=None, fetch=None,
               max_redirects: int = MAX_REDIRECTS) -> dict:
    """Validated GET that follows redirects, RE-CHECKING each hop (SSRF/rebinding/open-redirect).
    Returns RAW bytes so callers can decode text OR extract binary (PDF). Shared by web_fetch and
    web_extract so the boundary lives in one place."""
    from urllib.parse import urljoin
    fetch = fetch or raw_get
    ok, reason = check_url(url, allow_private, resolver)
    if not ok:
        return {"ok": False, "error": reason}
    chain, cur = [], url
    for _hop in range(max_redirects + 1):
        res = fetch(cur, TIMEOUT, MAX_BODY)
        if "error" in res:
            return {"ok": False, "error": res["error"], "url": cur, "redirects": chain}
        status = res["status"]
        chain.append({"url": cur, "status": status})
        if 300 <= status < 400:
            loc = res["headers"].get("Location") or res["headers"].get("location")
            if not loc:
                break
            nxt = urljoin(cur, loc)
            ok2, reason2 = check_url(nxt, allow_private, resolver)
            if not ok2:
                return {"ok": False, "error": f"redirect blocked: {reason2}", "redirects": chain}
            cur = nxt
            continue
        ctype = res["headers"].get("Content-Type", res["headers"].get("content-type", ""))
        return {"ok": status < 400, "status": status, "final_url": cur, "content_type": ctype,
                "raw": res["body"][:MAX_BODY], "truncated": len(res["body"]) > MAX_BODY,
                "headers": res["headers"], "redirects": chain}
    return {"ok": False, "error": f"too many redirects (> {max_redirects})", "redirects": chain}


def raw_get(url: str, timeout: int = TIMEOUT, max_bytes: int = MAX_BODY) -> dict:
    """One GET, NO auto-redirect (caller re-validates each hop). Returns status/headers/body or
    {'error': ...}. Never raises."""
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": UA, "Accept": "*/*"})

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        r = opener.open(req, timeout=timeout)
        return {"status": r.status, "headers": dict(r.headers), "body": r.read(max_bytes + 1)}
    except urllib.error.HTTPError as e:
        try:
            body = e.read(max_bytes + 1)
        except Exception:                                     # noqa: BLE001
            body = b""
        return {"status": e.code, "headers": dict(e.headers or {}), "body": body}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"error": str(e)[:300]}
