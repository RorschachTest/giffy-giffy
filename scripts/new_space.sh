#!/usr/bin/env bash
# Start a fresh data space: a new database and a new data folder, filled by
# processing COPIES of the original clips again with the current pipeline.
# Nothing in the old space is changed or deleted, so switching back is one edit.
#
#   scripts/new_space.sh v2            create database memeclip_v2 + folder data/v2,
#                                      carry over people, face photos and search history,
#                                      queue copies of the originals in data/v2/inbox
#   scripts/new_space.sh v2 --switch   the same, then point the app at it (.env) and restart
#
# Switch back: delete the DATABASE_URL and DATA_DIR lines from .env (a copy of the
# previous file is kept as .env.bak) and run: docker compose up -d api worker
set -euo pipefail
cd "$(dirname "$0")/.."

name="${1:?usage: scripts/new_space.sh <name> [--switch]}"
[[ "$name" =~ ^[a-z0-9_]+$ ]] || { echo "name must be lowercase letters, digits or _"; exit 1; }
db="memeclip_${name}"
dir="data/${name}"
url="postgresql://memeclip:memeclip@db:5432/${db}"
psql() { docker compose exec -T db psql -U memeclip -v ON_ERROR_STOP=1 "$@"; }

echo "1/5 database ${db}"
if psql -d memeclip -Atc "SELECT 1 FROM pg_database WHERE datname = '${db}'" | grep -q 1; then
  echo "    exists, kept as it is"
else
  psql -d memeclip -c "CREATE DATABASE ${db}" >/dev/null
fi

echo "2/5 tables"
docker compose run --rm -T --no-deps -e DATABASE_URL="$url" api python -c "from app import db; db.init_db()"

echo "3/5 people, face photos, search history (only into empty tables)"
for t in people face_refs query_log; do
  n=$(psql -d "$db" -Atc "SELECT count(*) FROM ${t}")
  if [ "$n" = "0" ]; then
    docker compose exec -T db pg_dump -U memeclip -d memeclip --data-only -t "$t" | psql -d "$db" -q >/dev/null
  fi
  echo "    ${t}: $(psql -d "$db" -Atc "SELECT count(*) FROM ${t}") rows"
done

echo "4/5 folder ${dir}"
mkdir -p "$dir"/{inbox,media,processed,failed,gallery}
cp -Rn data/gallery/. "$dir/gallery/" 2>/dev/null || true
for f in data/*.json data/*.txt; do [ -f "$f" ] && cp -n "$f" "$dir/" || true; done   # test queries, references
queued=0
for f in data/processed/*; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  if [ ! -e "$dir/inbox/$base" ] && [ ! -e "$dir/processed/$base" ]; then
    cp "$f" "$dir/inbox/$base"; queued=$((queued + 1))
  fi
done
echo "    queued ${queued} file(s) (copies; data/processed is untouched)"

if [ "${2:-}" = "--switch" ]; then
  echo "5/5 switching the app to ${db} + ${dir}"
  [ -f .env ] && cp .env .env.bak
  touch .env
  grep -v -E '^(DATABASE_URL|DATA_DIR)=' .env > .env.tmp || true
  printf 'DATABASE_URL=%s\nDATA_DIR=/%s\n' "$url" "$dir" >> .env.tmp
  mv .env.tmp .env
  docker compose up -d api worker
  echo "    done: the worker is processing ${dir}/inbox (docker compose logs -f worker)"
else
  echo "5/5 not switched; run again with --switch to use it"
fi
