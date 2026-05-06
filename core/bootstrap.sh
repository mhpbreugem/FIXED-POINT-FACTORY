#!/usr/bin/env bash
# DEPRECATED — superseded by the pure GitHub Actions architecture in
# .github/workflows/solve-tasks.yml. Kept here in case GCP spot VMs
# become useful again. Do not use for new runs.
#
# bootstrap.sh — VM startup script for fixed-point-factory solver VMs.
#
# Designed to run as a GCP startup script or via curl-pipe on a fresh VM.
# Installs dependencies, clones the repo, starts the heartbeat, then enters
# the worker loop.
#
# Required env (set via GCP metadata or export before running):
#   GITHUB_TOKEN        Personal access token with repo read/write
#   PROJECT             Which project to work on (e.g. REZN)
#   BRANCH              Git branch to use
#
# Optional env:
#   WORKER_ID           Defaults to hostname
#   MAX_RUN_HOURS       Exit after this many hours (default: unlimited)
#   REPO_URL            Defaults to https://github.com/mhpbreugem/fixed-point-factory.git
#   PYTHON_DEPS         Space-separated pip packages (default: mpmath numpy scipy)
#   INTERVAL            Heartbeat interval in seconds (default: 300)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/mhpbreugem/fixed-point-factory.git}"
PROJECT="${PROJECT:?PROJECT env var required}"
BRANCH="${BRANCH:?BRANCH env var required}"
WORKER_ID="${WORKER_ID:-$(hostname)}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-0}"   # 0 = unlimited
PYTHON_DEPS="${PYTHON_DEPS:-mpmath numpy scipy}"
REPO_DIR="/opt/fixed-point-factory"

export WORKER_ID PROJECT BRANCH REPO_DIR

echo "=== fixed-point-factory bootstrap ==="
echo "  WORKER_ID     : ${WORKER_ID}"
echo "  PROJECT       : ${PROJECT}"
echo "  BRANCH        : ${BRANCH}"
echo "  MAX_RUN_HOURS : ${MAX_RUN_HOURS}"
echo "  REPO_URL      : ${REPO_URL}"
echo "======================================"

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq git python3 python3-pip python3-venv curl
elif command -v yum &>/dev/null; then
    yum install -y git python3 python3-pip
fi

# ---------------------------------------------------------------------------
# 2. Clone / update repo
# ---------------------------------------------------------------------------
AUTHED_URL="https://${GITHUB_TOKEN}@${REPO_URL#https://}"

if [ -d "${REPO_DIR}/.git" ]; then
    echo "[bootstrap] pulling latest..."
    cd "${REPO_DIR}"
    git remote set-url origin "${AUTHED_URL}"
    git fetch origin "${BRANCH}"
    git checkout "${BRANCH}"
    git pull --rebase origin "${BRANCH}"
else
    echo "[bootstrap] cloning..."
    git clone --branch "${BRANCH}" "${AUTHED_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"
fi

# Store credentials for subsequent pushes
git config credential.helper store
echo "https://${GITHUB_TOKEN}@github.com" > ~/.git-credentials
git config user.email "worker@fixed-point-factory"
git config user.name "${WORKER_ID}"

export REPO_ROOT="${REPO_DIR}"

# ---------------------------------------------------------------------------
# 3. Python environment
# ---------------------------------------------------------------------------
VENV="${REPO_DIR}/.venv"
if [ ! -d "${VENV}" ]; then
    python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Install dependencies for the project
pip install --quiet ${PYTHON_DEPS}

# Install any project-specific requirements
PROJ_REQS="${REPO_DIR}/projects/${PROJECT}/requirements.txt"
if [ -f "${PROJ_REQS}" ]; then
    pip install --quiet -r "${PROJ_REQS}"
fi

# ---------------------------------------------------------------------------
# 4. Start heartbeat in background
# ---------------------------------------------------------------------------
export INTERVAL="${INTERVAL:-300}"
bash "${REPO_DIR}/core/heartbeat.sh" >> /var/log/heartbeat.log 2>&1 &
HEARTBEAT_PID=$!
echo "[bootstrap] heartbeat PID=${HEARTBEAT_PID}"

# ---------------------------------------------------------------------------
# 5. Worker loop
# ---------------------------------------------------------------------------
START_TS=$(date +%s)

cleanup() {
    echo "[bootstrap] cleaning up..."
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    # Release any claim this worker holds
    cd "${REPO_DIR}"
    python3 core/claim_task.py release \
        --project "${PROJECT}" \
        --worker-id "${WORKER_ID}" \
        --branch "${BRANCH}" 2>/dev/null || true
    echo "[bootstrap] done."
}
trap cleanup EXIT INT TERM

PROJECT_SOLVER="${REPO_DIR}/projects/${PROJECT}/solver_code/solve.py"

while true; do
    # Check elapsed time
    if [ "${MAX_RUN_HOURS}" -gt 0 ] 2>/dev/null; then
        ELAPSED=$(( $(date +%s) - START_TS ))
        ELAPSED_HOURS=$(echo "scale=2; ${ELAPSED}/3600" | bc)
        if (( $(echo "${ELAPSED_HOURS} >= ${MAX_RUN_HOURS}" | bc -l) )); then
            echo "[bootstrap] MAX_RUN_HOURS=${MAX_RUN_HOURS} elapsed, exiting."
            break
        fi
    fi

    # Pull latest
    cd "${REPO_DIR}"
    git pull --rebase origin "${BRANCH}" --quiet 2>/dev/null || true

    # Clean up orphan progress files from crashed prior workers.
    python3 - <<'PY' || true
import json, os
from pathlib import Path
proj = os.environ["PROJECT"]
root = Path(os.environ.get("REPO_DIR", "."))
queue = json.load(open(root / "projects" / proj / "TASK_QUEUE.json"))
claimed_ids = {t["id"] for t in queue["tasks"] if t.get("status") == "claimed"}
pdir = root / "projects" / proj / "progress"
if pdir.is_dir():
    for f in pdir.glob("*.json"):
        if f.stem not in claimed_ids:
            f.unlink()
PY
    git add -A "projects/${PROJECT}/progress/" 2>/dev/null || true
    git diff --cached --quiet 2>/dev/null || git commit -m "cleanup orphan progress" --quiet 2>/dev/null || true
    git push origin "${BRANCH}" --quiet 2>/dev/null || true

    # Release stale claims
    python3 core/claim_task.py release \
        --project "${PROJECT}" \
        --branch "${BRANCH}" \
        --max-age-hours 6 2>/dev/null || true

    # Find and claim a task
    TASK_ID=$(python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, os.environ["REPO_ROOT"])
from core.claim_task import find_ready_task
t = find_ready_task(os.environ["PROJECT"], os.environ.get("WORKER_ID"))
print(t["id"] if t else "")
PYEOF
)

    if [ -z "${TASK_ID}" ]; then
        echo "[bootstrap] no ready tasks, sleeping 60s..."
        sleep 60
        continue
    fi

    echo "[bootstrap] trying to claim ${TASK_ID}..."
    if ! python3 core/claim_task.py claim \
            --project "${PROJECT}" \
            --task-id "${TASK_ID}" \
            --worker-id "${WORKER_ID}" \
            --branch "${BRANCH}"; then
        echo "[bootstrap] claim failed (race), retrying..."
        continue
    fi

    echo "[bootstrap] claimed ${TASK_ID}, starting solver..."
    if [ -f "${PROJECT_SOLVER}" ]; then
        python3 "${PROJECT_SOLVER}" \
            --project "${PROJECT}" \
            --task-id "${TASK_ID}" \
            --branch "${BRANCH}" \
            --worker-id "${WORKER_ID}" || true
    else
        echo "[bootstrap] ERROR: solver not found at ${PROJECT_SOLVER}"
        python3 core/claim_task.py bail \
            --project "${PROJECT}" \
            --task-id "${TASK_ID}" \
            --reason "solver_code/solve.py not found" \
            --branch "${BRANCH}"
    fi
done
