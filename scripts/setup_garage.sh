#!/usr/bin/env bash
# Set up the local Garage bucket for the clips, and (with --switch) serve from it.
#
#   scripts/setup_garage.sh            start Garage, create the bucket and an access key,
#                                      copy the existing clips in (the local files are kept)
#   scripts/setup_garage.sh --switch   the same, then STORAGE=s3 in .env and restart api + worker
#
# Secrets (Garage's RPC secret and admin token, the bucket key) are generated here and
# written to .env only; nothing is printed. Safe to run again: every step checks first.
# Switch back: set STORAGE=local in .env (or delete the line), docker compose up -d api worker.
set -euo pipefail
cd "$(dirname "$0")/.."

BUCKET="${S3_BUCKET:-memeclip}"
KEY_NAME="memeclip-app"

touch .env
getenv() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }
setenv() {   # KEY VALUE -> .env, replacing an existing line; the value is never echoed
  grep -v -E "^$1=" .env > .env.tmp || true
  printf '%s=%s\n' "$1" "$2" >> .env.tmp
  mv .env.tmp .env
}
g() { docker compose exec -T garage /garage "$@"; }

echo "1/6 secrets in .env"
[ -n "$(getenv GARAGE_RPC_SECRET)" ] || setenv GARAGE_RPC_SECRET "$(openssl rand -hex 32)"
[ -n "$(getenv GARAGE_ADMIN_TOKEN)" ] || setenv GARAGE_ADMIN_TOKEN "$(openssl rand -hex 32)"
profiles="$(getenv COMPOSE_PROFILES)"
case ",${profiles}," in *,bucket,*) ;; *) setenv COMPOSE_PROFILES "${profiles:+${profiles},}bucket" ;; esac

echo "2/6 starting garage"
docker compose up -d garage
for _ in $(seq 60); do g status >/dev/null 2>&1 && break; sleep 1; done
g status >/dev/null

echo "3/6 layout (one node)"
if g status | grep -q "NO ROLE ASSIGNED"; then
  node="$(g node id -q | cut -d@ -f1)"
  g layout assign -z dc1 -c 1G "$node" >/dev/null
  version="$(g layout show | sed -n 's/.*apply --version \([0-9]*\).*/\1/p' | head -1)"
  g layout apply --version "${version:-1}" >/dev/null
fi

echo "4/6 bucket ${BUCKET} and key ${KEY_NAME}"
g bucket info "$BUCKET" >/dev/null 2>&1 || g bucket create "$BUCKET" >/dev/null
g key info "$KEY_NAME" >/dev/null 2>&1 || g key create "$KEY_NAME" >/dev/null
g bucket allow --read --write --owner "$BUCKET" --key "$KEY_NAME" >/dev/null
info="$(g key info --show-secret "$KEY_NAME")"
key_id="$(printf '%s\n' "$info" | sed -n 's/^Key ID: *//p' | head -1)"
secret="$(printf '%s\n' "$info" | sed -n 's/^Secret key: *//p' | head -1)"
[ -n "$key_id" ] && [ -n "$secret" ] || { echo "could not read the key from 'garage key info'"; exit 1; }
setenv S3_ACCESS_KEY "$key_id"
setenv S3_SECRET_KEY "$secret"

echo "5/6 copying existing clips into the bucket (local files kept)"
docker compose build -q api worker
docker compose run --rm -T --no-deps -e STORAGE=s3 api python -m app.storage --upload-all

if [ "${1:-}" = "--switch" ]; then
  echo "6/6 serving clips from the bucket"
  setenv STORAGE s3
  docker compose up -d --no-deps api worker
else
  echo "6/6 not switched; run again with --switch to serve from the bucket"
fi
