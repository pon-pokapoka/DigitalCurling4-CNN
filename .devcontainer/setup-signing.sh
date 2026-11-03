#!/usr/bin/env bash
# Configure SSH commit/tag signing inside the dev container.
#
# The private key stays on the host: the host's ssh-agent socket is bind-mounted
# into the container (see devcontainer.json), and git signs through it. Settings
# are written to GIT_CONFIG_GLOBAL (the dc4-cnn-git volume), never to the repo.
#
#   .devcontainer/setup-signing.sh            # use the only key in the agent
#   .devcontainer/setup-signing.sh <pattern>  # pick the agent key whose line matches <pattern>
#   .devcontainer/setup-signing.sh --check    # verify signing works (run on container start)
set -euo pipefail

die()  { echo "setup-signing: $*" >&2; exit 1; }
warn() { echo "setup-signing: WARNING: $*" >&2; }

agent_keys() {
  [[ -n "${SSH_AUTH_SOCK:-}" && -S "$SSH_AUTH_SOCK" ]] || return 2
  ssh-add -L 2>/dev/null | grep -E '^(ssh-|ecdsa-|sk-)' || return 1
}

check() {
  local key
  if [[ "$(git config --global --get commit.gpgsign || true)" != "true" ]]; then
    warn "commit signing is not configured. Run: .devcontainer/setup-signing.sh"
    return 0
  fi
  key="$(git config --global --get user.signingkey || true)"
  key="${key#key::}"
  if ! agent_keys >/dev/null; then
    warn "no ssh-agent keys reachable (SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-unset}); commits will fail until the host agent is forwarded. Recreate the container with .devcontainer/dev.sh after 'ssh-add' on the host."
    return 0
  fi
  if ! agent_keys | awk '{print $1" "$2}' | grep -qxF "$key"; then
    warn "the configured signing key is not loaded in the forwarded ssh-agent; run 'ssh-add' for it on the host."
    return 0
  fi
  echo "setup-signing: OK, commits will be signed with $(ssh-keygen -lf <(echo "$key") | awk '{print $2}')"
}

if [[ "${1:-}" == "--check" ]]; then check; exit 0; fi

[[ -n "${GIT_CONFIG_GLOBAL:-}" ]] || die "GIT_CONFIG_GLOBAL is not set; run this inside the dev container."
keys="$(agent_keys)" || die "no keys in a forwarded ssh-agent (SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-unset}). On the host: ssh-add <key>, then recreate the container: .devcontainer/dev.sh down && .devcontainer/dev.sh up"

if [[ -n "${1:-}" ]]; then
  keys="$(grep -F -- "$1" <<<"$keys" || true)"
  [[ -n "$keys" ]] || die "no agent key matches '$1'. Available:"$'\n'"$(ssh-add -l)"
fi
[[ "$(wc -l <<<"$keys")" -eq 1 ]] || die "several keys in the agent; pass a pattern (part of the comment) to pick one:"$'\n'"$keys"

key="$(awk '{print $1" "$2}' <<<"$keys")"
email="$(git config --get user.email || true)"
[[ -n "$email" ]] || die "user.email is not set; set it first: git config --global user.email you@example.com"

mkdir -p "$(dirname "$GIT_CONFIG_GLOBAL")"
allowed="$(dirname "$GIT_CONFIG_GLOBAL")/allowed_signers"
touch "$allowed"
grep -qF -- "$key" "$allowed" || printf '%s namespaces="git" %s\n' "$email" "$key" >> "$allowed"

git config --global gpg.format ssh
git config --global user.signingkey "key::$key"
git config --global gpg.ssh.allowedSignersFile "$allowed"
git config --global commit.gpgsign true
git config --global tag.gpgsign true

# Prove the agent can actually sign with this key.
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
echo "$key" > "$tmp/key.pub"
echo test > "$tmp/msg"
ssh-keygen -Y sign -n git -f "$tmp/key.pub" -U "$tmp/msg" >/dev/null 2>&1 \
  || die "the agent refused to sign (key needs confirmation, or a hardware key needs a touch?)"

echo "setup-signing: commits and tags will now be signed with $(ssh-keygen -lf "$tmp/key.pub" | awk '{print $2}')"
echo "If you haven't yet, register this public key on GitHub as a *signing* key so commits show 'Verified':"
echo "  gh auth refresh -s admin:ssh_signing_key && echo '$key' | gh ssh-key add --type signing --title dc4-cnn-devcontainer"
echo "  (or GitHub > Settings > SSH and GPG keys > New SSH key > Key type: Signing Key)"
