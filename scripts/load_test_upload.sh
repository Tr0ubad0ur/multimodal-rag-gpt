#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
TOKEN="${TOKEN:-}"
FILE_PATH="${FILE_PATH:-}"
REQUESTS="${REQUESTS:-50}"
PARALLEL="${PARALLEL:-10}"
ENDPOINT="${ENDPOINT:-/files/upload}"

if [[ -z "$TOKEN" ]]; then
  echo "TOKEN is required"
  exit 1
fi

if [[ -z "$FILE_PATH" || ! -f "$FILE_PATH" ]]; then
  echo "FILE_PATH must point to existing file"
  exit 1
fi

echo "base_url=$BASE_URL endpoint=$ENDPOINT requests=$REQUESTS parallel=$PARALLEL"

seq "$REQUESTS" | xargs -I{} -P "$PARALLEL" bash -c '
  code=$(curl -sS -o /dev/null -w "%{http_code}" \
    -X POST "'"$BASE_URL$ENDPOINT"'" \
    -H "Authorization: Bearer '"$TOKEN"'" \
    -F "file=@'"$FILE_PATH"'")
  echo "$code"
' | sort | uniq -c
