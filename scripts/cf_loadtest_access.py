#!/usr/bin/env python3
"""Let this machine's IP skip memeclip's Cloudflare rate limit while load testing, then undo it.

    python3 scripts/cf_loadtest_access.py status    show the rule (if any) and this machine's IPs
    python3 scripts/cf_loadtest_access.py apply     add/refresh the rule for this machine's current IPs
    python3 scripts/cf_loadtest_access.py remove    delete the rule (run this when the test is done)

What it changes: one custom WAF rule on the aiwroteit.dev zone, named RULE_NAME below,
that SKIPS the rate-limiting phase for requests to memeclip.aiwroteit.dev from this
machine's IPv4 address and IPv6 /64 network. Nothing else is skipped: DDoS protection,
the Access policies and the app's own sign-in all still apply. The rate-limit rule itself
("memeclip: limit requests per visitor") is never touched.

Needs, in .env (never printed):
    CLOUDFLARE_API_TOKEN   an API token limited to the aiwroteit.dev zone with
                           "Zone > Zone WAF > Edit" and "Zone > Zone > Read"
    CLOUDFLARE_ZONE_ID     optional; looked up from CF_ZONE_NAME when missing
"""
from __future__ import annotations

import ipaddress
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CF_ZONE_NAME = "aiwroteit.dev"
HOST = "memeclip.aiwroteit.dev"
RULE_NAME = "memeclip: owner IP skips rate limit (load testing)"
API = "https://api.cloudflare.com/client/v4"
PHASE = "http_request_firewall_custom"


def env() -> dict[str, str]:
    path = Path(__file__).resolve().parent.parent / ".env"
    out = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def call(method: str, path: str, token: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read() or b"{}")
        if e.code == 404 and path.endswith("/entrypoint"):
            return {"result": None}
        msgs = "; ".join(x.get("message", "") for x in detail.get("errors", [])) or e.reason
        raise SystemExit(f"Cloudflare API {method} {path.split('?')[0]} failed ({e.code}): {msgs}")


def my_ips() -> list[str]:
    """This machine's public IPs as Cloudflare sees them: the IPv4 address and the IPv6 /64."""
    out = []
    for family in (4, 6):
        try:
            # Cloudflare's own trace endpoint, on the memeclip hostname: the address that reaches memeclip
            import socket
            orig = socket.getaddrinfo
            socket.getaddrinfo = lambda *a, **k: [x for x in orig(*a, **k)
                                                    if x[0] == (socket.AF_INET if family == 4 else socket.AF_INET6)]
            try:
                with urllib.request.urlopen(f"https://{HOST}/cdn-cgi/trace", timeout=8) as r:
                    text = r.read().decode()
            finally:
                socket.getaddrinfo = orig
        except Exception:
            continue
        ip = next((l[3:] for l in text.splitlines() if l.startswith("ip=")), None)
        if not ip:
            continue
        addr = ipaddress.ip_address(ip)
        out.append(str(addr) if addr.version == 4 else str(ipaddress.ip_network(f"{ip}/64", strict=False)))
    if not out:
        raise SystemExit(f"could not find this machine's public IP (is {HOST} reachable?)")
    return out


def zone_id(token: str, e: dict) -> str:
    if e.get("CLOUDFLARE_ZONE_ID"):
        return e["CLOUDFLARE_ZONE_ID"]
    res = call("GET", f"/zones?name={CF_ZONE_NAME}", token).get("result") or []
    if not res:
        raise SystemExit(f"zone {CF_ZONE_NAME} not visible to this token")
    return res[0]["id"]


def entrypoint(token: str, zone: str) -> dict | None:
    return call("GET", f"/zones/{zone}/rulesets/phases/{PHASE}/entrypoint", token).get("result")


def ours(ruleset: dict | None) -> list[dict]:
    return [r for r in (ruleset or {}).get("rules") or [] if r.get("description") == RULE_NAME]


def rule_body(ips: list[str]) -> dict:
    return {
        "description": RULE_NAME,
        "expression": f'(http.host eq "{HOST}" and ip.src in {{{" ".join(ips)}}})',
        "action": "skip",
        "action_parameters": {"phases": ["http_ratelimit"]},   # skip ONLY rate limiting
        "logging": {"enabled": True},
        "enabled": True,
    }


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "status"
    if cmd not in ("status", "apply", "remove"):
        print(__doc__)
        return 2
    e = env()
    token = e.get("CLOUDFLARE_API_TOKEN") or ""
    if not token:
        raise SystemExit("CLOUDFLARE_API_TOKEN is not set in .env (see the header of this script)")
    zone = zone_id(token, e)
    rs = entrypoint(token, zone)
    existing = ours(rs)

    if cmd == "status":
        print(f"this machine: {', '.join(my_ips())}")
        if not existing:
            print("rule: not present (rate limit applies to everyone)")
        for r in existing:
            print(f"rule: {'ENABLED' if r.get('enabled') else 'disabled'}  {r['expression']}  "
                  f"(last updated {r.get('last_updated', '?')})")
        return 0

    if cmd == "remove":
        if not existing:
            print("nothing to remove: the rule is not present")
            return 0
        for r in existing:
            call("DELETE", f"/zones/{zone}/rulesets/{rs['id']}/rules/{r['id']}", token)
        print(f"removed {len(existing)} rule(s); the rate limit applies to everyone again")
        return 0

    # apply: one rule, refreshed with today's IPs
    ips = my_ips()
    body = rule_body(ips)
    if rs is None:
        call("PUT", f"/zones/{zone}/rulesets/phases/{PHASE}/entrypoint", token, {"rules": [body]})
    elif existing:
        call("PATCH", f"/zones/{zone}/rulesets/{rs['id']}/rules/{existing[0]['id']}", token, body)
        for extra in existing[1:]:
            call("DELETE", f"/zones/{zone}/rulesets/{rs['id']}/rules/{extra['id']}", token)
    else:
        call("POST", f"/zones/{zone}/rulesets/{rs['id']}/rules", token,
             {**body, "position": {"index": 1}})   # first, so it runs before anything else
    print(f"applied at {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}: {HOST} rate limit skipped for {', '.join(ips)}")
    print("remember: python3 scripts/cf_loadtest_access.py remove   when the test is done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
