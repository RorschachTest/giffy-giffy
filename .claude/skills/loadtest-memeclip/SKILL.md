---
name: loadtest-memeclip
description: Load test memeclip on prod safely. Lets this machine's IP skip the Cloudflare rate limit for the test (apply), runs the test, then removes the exemption (remove). Use when the user says "load test memeclip", "start/stop load testing", "whitelist my IP for testing", or asks to undo the load-test exemption.
---

# Load testing memeclip on prod

Prod (`memeclip.aiwroteit.dev`) has a Cloudflare rate limit: more than 100 requests in
10 s from one IP is blocked for 10 s ("memeclip: limit requests per visitor"). A load test
from this machine trips it, so for the test only, one extra rule lets this machine's IPs
**skip the rate limit** (and nothing else). The skill always ends by removing it.

Everything goes through `scripts/cf_loadtest_access.py` (Cloudflare API). It needs in `.env`:
- `CLOUDFLARE_API_TOKEN`: a token limited to the `aiwroteit.dev` zone with
  **Zone › Zone WAF › Edit** and **Zone › Zone › Read** (create it in Cloudflare → My Profile →
  API Tokens → Create Token → Custom token). Never print or echo it.
- `CLOUDFLARE_ZONE_ID`: optional (looked up when missing).

If the token is missing, stop and tell the user how to create it (above). Do not fall back
to clicking the dashboard to add a security exemption without the user's explicit go-ahead.

## Steps

1. **Check** what is there now:
   ```bash
   python3 scripts/cf_loadtest_access.py status
   ```
2. **Apply** the exemption for this machine's current IPs (IPv4 address + IPv6 /64; home IPs
   change, so always re-apply instead of reusing an old rule):
   ```bash
   python3 scripts/cf_loadtest_access.py apply
   ```
   Confirm it works: 150 quick requests must all return 200.
   ```bash
   seq 150 | xargs -P 20 -I{} curl -s -o /dev/null -w "%{http_code}\n" https://memeclip.aiwroteit.dev/privacy | sort | uniq -c
   ```
3. **Run the load test** the user asked for. Prod is the user's Mac (API, Postgres, videos):
   ramp up in steps (e.g. 10 → 25 → 50 concurrent), watch `docker stats` and response times,
   and stop at the first sign of errors or the Mac struggling. Signed-in endpoints (`/search`)
   need the user's `mc_session` cookie: ask them to paste it for the test, use it only in the
   test command, and never store it in a file.
4. **Remove** the exemption as soon as the test is done, also if it failed or was stopped:
   ```bash
   python3 scripts/cf_loadtest_access.py remove
   python3 scripts/cf_loadtest_access.py status
   ```
   `status` must say "rule: not present". Tell the user the exemption is gone.

## Notes
- The rule only skips **rate limiting** for `memeclip.aiwroteit.dev` from these IPs. DDoS
  protection, Cloudflare Access policies and memeclip's Google sign-in still apply.
- The rule's name is `memeclip: owner IP skips rate limit (load testing)`; `remove` deletes
  every rule with that name, including one created by hand in the dashboard.
- Do not change or disable the rate-limit rule itself for a load test.
