#!/usr/bin/env bash
# create_gcp_vm.sh — Spin up a GCP spot VM that runs a fixed-point-factory worker.
#
# Usage:
#   GITHUB_TOKEN=ghp_xxx PROJECT=REZN BRANCH=main VM_NAME=solver-1 bash core/create_gcp_vm.sh
#
# Required env:
#   GITHUB_TOKEN     GitHub PAT with repo read/write
#   PROJECT          Which project (e.g. REZN)
#   BRANCH           Git branch
#   VM_NAME          Name for the GCP instance (e.g. solver-1)
#
# Optional env:
#   GCP_PROJECT      GCP project ID (default: current gcloud project)
#   GCP_ZONE         GCP zone (default: us-central1-a)
#   MACHINE_TYPE     GCP machine type (default: n2-standard-4)
#   MAX_RUN_HOURS    Auto-exit after N hours (default: 1; GCP max-run-duration also set to 1h)
#   DISK_SIZE        Boot disk size in GB (default: 20)
#   REPO_URL         Repo URL (default: https://github.com/mhpbreugem/fixed-point-factory.git)

set -euo pipefail

# ---------------------------------------------------------------------------
# Required
# ---------------------------------------------------------------------------
: "${GITHUB_TOKEN:?GITHUB_TOKEN env var required}"
: "${PROJECT:?PROJECT env var required}"
: "${BRANCH:?BRANCH env var required}"
: "${VM_NAME:?VM_NAME env var required}"

# ---------------------------------------------------------------------------
# Optional with defaults
# ---------------------------------------------------------------------------
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-n2-standard-4}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-1}"
DISK_SIZE="${DISK_SIZE:-20}"
REPO_URL="${REPO_URL:-https://github.com/mhpbreugem/fixed-point-factory.git}"

echo "=== Creating GCP spot VM ==="
echo "  VM_NAME       : ${VM_NAME}"
echo "  PROJECT       : ${PROJECT}"
echo "  BRANCH        : ${BRANCH}"
echo "  GCP_PROJECT   : ${GCP_PROJECT}"
echo "  GCP_ZONE      : ${GCP_ZONE}"
echo "  MACHINE_TYPE  : ${MACHINE_TYPE}"
echo "  MAX_RUN_HOURS : ${MAX_RUN_HOURS}"
echo "============================"

# Build startup script that exports env vars and runs bootstrap.sh
STARTUP_SCRIPT=$(cat <<STARTUP
#!/bin/bash
export GITHUB_TOKEN="${GITHUB_TOKEN}"
export PROJECT="${PROJECT}"
export BRANCH="${BRANCH}"
export WORKER_ID="${VM_NAME}"
export MAX_RUN_HOURS="${MAX_RUN_HOURS}"
export REPO_URL="${REPO_URL}"
bash /tmp/bootstrap.sh >> /var/log/solver.log 2>&1
STARTUP
)

# Use a metadata-based startup: fetch bootstrap.sh from the repo, then run it.
# We embed the bootstrap URL so the VM can pull the latest version.
STARTUP_FULL=$(cat <<STARTUP_FULL
#!/bin/bash
export GITHUB_TOKEN="${GITHUB_TOKEN}"
export PROJECT="${PROJECT}"
export BRANCH="${BRANCH}"
export WORKER_ID="${VM_NAME}"
export MAX_RUN_HOURS="${MAX_RUN_HOURS}"
export REPO_URL="${REPO_URL}"

# Install git first (needed to clone)
apt-get update -qq && apt-get install -y -qq git python3 python3-pip

# Clone repo to get bootstrap.sh
git clone --branch "${BRANCH}" \
    "https://${GITHUB_TOKEN}@${REPO_URL#https://}" \
    /opt/fixed-point-factory

# Run bootstrap
export REPO_ROOT=/opt/fixed-point-factory
bash /opt/fixed-point-factory/core/bootstrap.sh >> /var/log/solver.log 2>&1
STARTUP_FULL
)

gcloud compute instances create "${VM_NAME}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --max-run-duration="${MAX_RUN_HOURS}h" \
    --boot-disk-size="${DISK_SIZE}GB" \
    --boot-disk-type=pd-ssd \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --no-address \
    --metadata="startup-script=${STARTUP_FULL}"

echo ""
echo "VM ${VM_NAME} created. To monitor:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} -- tail -f /var/log/solver.log"
echo ""
echo "To delete (if not auto-deleted by max-run-duration):"
echo "  gcloud compute instances delete ${VM_NAME} --zone=${GCP_ZONE}"
