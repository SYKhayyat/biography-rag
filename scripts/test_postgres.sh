#!/usr/bin/env bash
# Run the Postgres integration tests against a throwaway local Postgres + pgvector (needs nix).
set -euo pipefail
cd "$(dirname "$0")/.."
exec nix-shell -p 'postgresql_16.withPackages (p: [ p.pgvector ])' --run '
  set -euo pipefail
  dir=$(mktemp -d /tmp/ragpg.XXXXXX)  # short path: unix socket paths max out at 107 bytes
  trap "pg_ctl -D $dir/data -m immediate stop >/dev/null 2>&1 || true; rm -rf $dir" EXIT
  initdb -D "$dir/data" -U postgres -A trust >/dev/null
  pg_ctl -D "$dir/data" -o "-k $dir -c listen_addresses= -p 55432" -l "$dir/log" -w start >/dev/null || { cat "$dir/log"; exit 1; }
  createdb -h "$dir" -p 55432 -U postgres rag_test
  export TEST_DATABASE_URL="postgresql://postgres@/rag_test?host=$dir&port=55432"
  nix-shell shell.nix --run "uv run pytest -q tests/test_postgres.py"
'
