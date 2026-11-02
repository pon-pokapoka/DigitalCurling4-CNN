#!/usr/bin/env bash
# Thin wrapper around the devcontainer CLI (npm i -g @devcontainers/cli).
#
#   .devcontainer/dev.sh up        # build image (if needed) and start the container
#   .devcontainer/dev.sh shell     # interactive shell inside the container
#   .devcontainer/dev.sh build     # cmake --build build inside the container
#   .devcontainer/dev.sh exec CMD  # run an arbitrary command inside the container
#   .devcontainer/dev.sh rebuild   # rebuild image from scratch and recreate container
#   .devcontainer/dev.sh down      # stop and remove the container
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="dc4-cnn-dev"   # keep in sync with runArgs in devcontainer.json

if command -v devcontainer >/dev/null 2>&1; then
  DC=(devcontainer)
elif command -v npx >/dev/null 2>&1; then
  DC=(npx --yes @devcontainers/cli)
else
  echo "devcontainer CLI not found. Install with: npm install -g @devcontainers/cli" >&2
  exit 1
fi

dc() { "${DC[@]}" "$@" --workspace-folder "$REPO_ROOT"; }
dc_exec() { "${DC[@]}" exec --workspace-folder "$REPO_ROOT" "$@"; }

cmd="${1:-shell}"; shift || true
case "$cmd" in
  up)      dc up ;;
  shell)   dc up >/dev/null && dc_exec bash -l ;;
  build)   dc up >/dev/null && dc_exec cmake --build build "$@" ;;
  exec)    dc up >/dev/null && dc_exec "$@" ;;
  rebuild) "${DC[@]}" up --workspace-folder "$REPO_ROOT" --remove-existing-container --build-no-cache ;;
  down)    docker rm -f "$CONTAINER_NAME" ;;
  *)       sed -n '2,10p' "$0"; exit 1 ;;
esac
