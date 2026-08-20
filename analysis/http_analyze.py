"""http_analyze.py — read-only security analysis of an HTTP response (headers + body). Pure.

Given a response an authorized fetch already returned, check the security-relevant properties a web
assessor looks at: missing security headers, insecure cookie flags, technology/version disclosure,
permissive CORS, and body info-disclosure — plus the form/parameter attack surface. Every result is
OBSERVED and a LEAD (a missing header is a weakness to confirm in context, not an exploit).
"""
from __future__ import annotations

import re

_SEC_HEADERS = {
    "content-security-policy": ("MEDIUM", "no Content-Security-Policy (XSS/injection mitigation)"),
    "x-frame-options": ("LOW", "no X-Frame-Options / frame-ancestors (clickjacking)"),
    "x-content-type-options": ("LOW", "no X-Content-Type-Options: nosniff (MIME sniffing)"),
    "referrer-policy": ("LOW", "no Referrer-Policy (referrer leakage)"),
}


def _lower(headers: dict) -> dict:
    return {str(k).lower(): str(v) for k, v in (headers or {}).items()}


def analyze_response(status, headers: dict, body: str, url: str = "") -> list[dict]:
    h = _lower(headers)
    body = body or ""
    https = str(url).lower().startswith("https")
    out = []

    for name, (sev, why) in _SEC_HEADERS.items():
        if name not in h:
            out.append({"issue": f"missing_{name}", "severity": sev, "why": why})
    if https and "strict-transport-security" not in h:
        out.append({"issue": "missing_hsts", "severity": "MEDIUM",
                    "why": "HTTPS without Strict-Transport-Security (downgrade/strip risk)"})

    if "server" in h and re.search(r"\d", h["server"]):
        out.append({"issue": "server_version_disclosure", "severity": "LOW",
                    "why": f"Server header discloses software/version: {h['server']}"})
    if "x-powered-by" in h:
        out.append({"issue": "tech_disclosure", "severity": "LOW",
                    "why": f"X-Powered-By discloses technology: {h['x-powered-by']}"})

    cookie = h.get("set-cookie", "")
    if cookie:
        cl = cookie.lower()
        if "httponly" not in cl:
            out.append({"issue": "cookie_no_httponly", "severity": "MEDIUM",
                        "why": "Set-Cookie without HttpOnly (JS can read it — XSS theft)"})
        if https and "secure" not in cl:
            out.append({"issue": "cookie_no_secure", "severity": "MEDIUM",
                        "why": "Set-Cookie without Secure over HTTPS (sent on cleartext too)"})
        if "samesite" not in cl:
            out.append({"issue": "cookie_no_samesite", "severity": "LOW",
                        "why": "Set-Cookie without SameSite (CSRF surface)"})

    acao = h.get("access-control-allow-origin", "")
    if acao == "*":
        creds = h.get("access-control-allow-credentials", "").lower() == "true"
        out.append({"issue": "cors_wildcard", "severity": "HIGH" if creds else "MEDIUM",
                    "why": "Access-Control-Allow-Origin: *" + (" with credentials (data theft)"
                                                               if creds else "")})

    for pat, why in [(r"(?i)\bSQL syntax\b|\bORA-\d{5}\b|\bSQLSTATE\b", "SQL error / DB info disclosure"),
                     (r"(?i)Traceback \(most recent call last\)|Exception in thread", "stack trace disclosed"),
                     (r"(?i)<title>\s*Index of /", "directory listing enabled"),
                     (r"(?i)phpinfo\(\)|<title>phpinfo", "phpinfo() exposed")]:
        if re.search(pat, body):
            out.append({"issue": "info_disclosure", "severity": "MEDIUM", "why": why})
    return out


def attack_surface(body: str, url: str = "") -> dict:
    """Forms (action/method/inputs) + URL query params — the parameter attack surface to test."""
    body = body or ""
    forms = []
    for fm in re.finditer(r"(?is)<form\b([^>]*)>(.*?)</form>", body):
        attrs, inner = fm.group(1), fm.group(2)
        action = (re.search(r"""(?i)action\s*=\s*['"]?([^'"\s>]+)""", attrs) or [None, ""])[1]
        method = (re.search(r"""(?i)method\s*=\s*['"]?([^'"\s>]+)""", attrs) or [None, "GET"])[1]
        inputs = re.findall(r"""(?i)<input\b[^>]*\bname\s*=\s*['"]?([^'"\s>]+)""", inner)
        forms.append({"action": action, "method": method.upper(), "inputs": inputs})
    params = re.findall(r"[?&]([A-Za-z0-9_\-\[\]]+)=", url or "")
    return {"forms": forms, "url_params": sorted(set(params))}
