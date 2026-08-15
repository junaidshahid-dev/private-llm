# Web application security

> source: curated-reference | version: v1 | updated: 2026-08-14 | authored

## Cross-site scripting (XSS)

XSS is the execution of attacker-controlled script in a victim's browser in the context of a
trusted site. Three kinds: **reflected** (payload in the request is echoed into the response),
**stored** (payload persisted and served to other users), and **DOM-based** (client-side JS
writes untrusted input into a sink like `innerHTML`). The root cause is untrusted data reaching an
output context without correct encoding.

The correct defence is **context-aware output encoding** — HTML, attribute, JavaScript, URL, and
CSS contexts each need different escaping. Autoescaping template engines handle the common case;
never disable it on untrusted input. A **Content-Security-Policy** is defence-in-depth, not the
primary fix: `script-src 'self'` blocks inline `<script>` and `onerror=` handlers, so a plain
`alert(1)` will not execute even where reflection exists. But CSP is bypassable — via a JSONP
endpoint or a script gadget on an allowed host, a permissive `nonce`/`unsafe-inline`, or
data-exfiltration through non-script vectors (dangling markup, CSS, prefetch). So reflection behind
a CSP is still a real finding: the encoding bug must be fixed, and CSP hardened, not treated as a
complete mitigation.

## SQL injection

User input concatenated into a SQL query lets an attacker alter the query. `username=admin'--`
comments out the rest of a `WHERE` clause and can bypass authentication. Fix with **parameterised
queries / prepared statements** so data can never be parsed as SQL; ORMs do this by default. Input
validation and least-privilege DB accounts are secondary. NoSQL has an analogue: a JSON field whose
value is an object like `{"$ne": null}` becomes a query operator — coerce fields to the expected
scalar type before they reach the query.

## CSRF

Cross-Site Request Forgery tricks an authenticated browser into sending a state-changing request.
Defend with anti-CSRF tokens (synchroniser pattern) and `SameSite=Lax`/`Strict` cookies; verify the
`Origin`/`Referer` for sensitive actions.

## SSRF

Server-Side Request Forgery: the server fetches an attacker-supplied URL. The classic impact is
reaching internal services the attacker cannot, especially a cloud metadata endpoint. Covered in
`cloud_security.md`. Blacklist filters (blocking `169.254.169.254` or `localhost` strings) are weak
— see the bypasses there. Defend with an allow-list, resolve-then-check against private ranges, and
disable or require IMDSv2 for metadata.

## Access-control flaws (IDOR / BOLA)

Broken Object-Level Authorization: the app trusts a client-supplied object id without checking the
requester owns it. Test by authenticating as user A, then requesting user B's object id across
every verb (GET/PUT/DELETE). Fix by checking ownership server-side on every object access — never
rely on unguessable ids alone.
