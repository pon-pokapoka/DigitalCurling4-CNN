#!/usr/bin/env bash
# Thin wrapper around the devcontainer CLI (npm i -g @devcontainers/cli).
#
#   .devcontainer/dev.sh up        # build image (if needed) and start the container
#   .devcontainer/dev.sh shell     # interactive shell inside the container
#   .devcontainer/dev.sh build     # cmake --build build inside the container
#   .devcontainer/dev.sh exec CMD  # run an arbitrary command inside the container
#   .devcontainer/dev.sh rebuild   # rebuild image from scratch and recreate container
#   .devcontainer/dev.sh down      # stop and remove the container
#
# Personal options (SSH agent forwarding, Claude Code) are read from
# .devcontainer/local.env, which is not committed; see local.env.example.
# Without it you get a plain container that needs no SSH key, GitHub or Claude.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="dc4-cnn-dev"   # keep in sync with runArgs in devcontainer.json
AGENT_TARGET="/run/host-ssh-agent.sock"

LOCAL_ENV="$REPO_ROOT/.devcontainer/local.env"
if [[ -f "$LOCAL_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$LOCAL_ENV"
fi
# Read by devcontainer.json (${localEnv:...}) as an image build argument.
export DC4_CLAUDE_CODE="${DC4_CLAUDE_CODE:-0}"
DC4_SSH_AGENT="${DC4_SSH_AGENT:-0}"

if command -v devcontainer >/dev/null 2>&1; then
  DC=(devcontainer)
elif command -v npx >/dev/null 2>&1; then
  DC=(npx --yes @devcontainers/cli)
else
  echo "devcontainer CLI not found. Install with: npm install -g @devcontainers/cli" >&2
  exit 1
fi

# DC4_SSH_AGENT=1: bind-mount the host ssh-agent socket (for git over SSH and
# commit signing). The private key never enters the container. Mounts only
# apply when the container is created: after changing this, run down + up.
up_args=()
exec_args=()
if [[ "$DC4_SSH_AGENT" == 1 ]]; then
  if [[ -z "${SSH_AUTH_SOCK:-}" || ! -S "$SSH_AUTH_SOCK" ]]; then
    echo "DC4_SSH_AGENT=1 but there is no ssh-agent on the host (SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-unset})." >&2
    echo "Start one and load your key first:  eval \"\$(ssh-agent)\" && ssh-add" >&2
    exit 1
  fi
  ssh-add -l >/dev/null 2>&1 || echo "Warning: the host ssh-agent has no keys loaded; run ssh-add." >&2
  up_args+=(--mount "type=bind,source=$SSH_AUTH_SOCK,target=$AGENT_TARGET"
            --remote-env "SSH_AUTH_SOCK=$AGENT_TARGET")
  exec_args+=(--remote-env "SSH_AUTH_SOCK=$AGENT_TARGET")
fi

dc_up() { "${DC[@]}" up --workspace-folder "$REPO_ROOT" "${up_args[@]}" "$@"; }
dc_exec() { "${DC[@]}" exec --workspace-folder "$REPO_ROOT" "${exec_args[@]}" "$@"; }

cmd="${1:-shell}"; shift || true
case "$cmd" in
  up)      dc_up ;;
  shell)   dc_up >/dev/null && dc_exec bash -l ;;
  build)   dc_up >/dev/null && dc_exec cmake --build build "$@" ;;
  exec)    dc_up >/dev/null && dc_exec "$@" ;;
  rebuild) dc_up --remove-existing-container --build-no-cache ;;
  down)    docker rm -f "$CONTAINER_NAME" ;;
  *)       sed -n '2,13p' "$0"; exit 1 ;;
esac
