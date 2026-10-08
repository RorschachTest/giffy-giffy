#!/usr/bin/env bash
# One command to build, start and check the whole stack.
#
#     bash scripts/smoke_test.sh
#
# Put a few clips (and their .json sidecars) in data/inbox first. Everything
# printed here is also saved to data/smoke_test.log.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=data/smoke_test.log
exec > >(tee "$LOG") 2>&1

API=http://localhost:8000
step() { printf '\n== %s\n' "$*"; }
pending() { find data/inbox -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.mov' -o -name '*.webm' -o -name '*.mkv' -o -name '*.m4v' \) | wc -l | tr -d ' '; }
fail() {
  printf '\nSMOKE TEST FAILED: %s\n' "$*"
  step "last lines from the containers"
  docker compose ps
  docker compose logs --tail 60 api worker
  exit 1
}

step "memeclip smoke test, $(date)"
docker --version || fail "docker is not installed or not on the PATH"
docker compose version || fail "docker compose is not available"
echo "clips waiting in data/inbox: $(pending)"

step "build and start (the first build takes several minutes)"
docker compose up --build -d || fail "docker compose up did not succeed"

step "wait for the API (first start downloads the embedding model)"
for i in $(seq 1 240); do
  curl -fsS "$API/search?q=" >/dev/null 2>&1 && break
  [ "$i" = 240 ] && fail "the API did not answer within 20 minutes"
  sleep 5
done
echo "API is up"

step "wait for the worker to empty data/inbox (first clip downloads the speech and face models)"
for i in $(seq 1 360); do
  [ "$(pending)" = 0 ] && break
  [ "$i" = 360 ] && fail "clips were still waiting after 30 minutes"
  [ $((i % 12)) = 0 ] && echo "  still waiting: $(pending) clip(s) left"
  sleep 5
done
echo "inbox is empty"

step "report"
docker compose exec -T api python -m app.smoke
STATUS=$?

step "worker log (last 40 lines)"
docker compose logs --no-log-prefix --tail 40 worker

if [ "$STATUS" = 0 ]; then
  printf '\nSMOKE TEST PASSED. Open %s\n' "$API"
else
  printf '\nSMOKE TEST FAILED: see the report above\n'
fi
exit "$STATUS"
