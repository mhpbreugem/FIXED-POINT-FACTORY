#!/usr/bin/env bash
# heartbeat.sh — write a timestamp every 5 minutes, commit and push.
# Run in background from bootstrap.sh.
#
# Required env:
#   PROJECT    e.g. REZN
#   WORKER_ID  e.g. solver-1
#   BRANCH     git branch to push to
#
# Optional env:
#   REPO_ROOT  (default: git rev-parse --show-toplevel)
#   INTERVAL   seconds between beats (default: 300)

set -euo pipefail

PROJECT="${PROJECT:?PROJECT env var required}"
WORKER_ID="${WORKER_ID:?WORKER_ID env var required}"
BRANCH="${BRANCH:?BRANCH env var required}"
INTERVAL="${INTERVAL:-300}"

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || echo ".")}"
HEARTBEAT_DIR="${REPO_ROOT}/projects/${PROJECT}/heartbeats"
HEARTBEAT_FILE="${HEARTBEAT_DIR}/${WORKER_ID}.txt"

mkdir -p "${HEARTBEAT_DIR}"

echo "[heartbeat] starting for worker=${WORKER_ID} project=${PROJECT} branch=${BRANCH}"

while true; do
    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "${TS}" > "${HEARTBEAT_FILE}"

    cd "${REPO_ROOT}"
    git add "projects/${PROJECT}/heartbeats/${WORKER_ID}.txt" 2>/dev/null || true

    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -m "heartbeat ${WORKER_ID} ${TS}" --quiet || true
        # push with simple retry
        for attempt in 1 2 3; do
            if git push origin "${BRANCH}" --quiet 2>/dev/null; then
                break
            fi
            git pull --rebase origin "${BRANCH}" --quiet 2>/dev/null || true
            sleep $((attempt * 2))
        done
    fi

    sleep "${INTERVAL}"
done
