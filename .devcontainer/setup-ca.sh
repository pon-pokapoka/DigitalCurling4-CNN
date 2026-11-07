#!/usr/bin/env bash
# Trust a private CA in the dev container, for a TLS server that needs it.
#
# Runs on container start only when DC4_CA_CERT is set: dev.sh mounts the
# certificate named by DC4_CA_CERT in .devcontainer/local.env and sets it to the
# path inside the container. It never stops the container from starting.
set -uo pipefail

dest=/usr/local/share/ca-certificates/dc4-ca.crt
warn() { echo "setup-ca: WARNING: $*" >&2; exit 0; }

cert="${DC4_CA_CERT:-}"
[[ -n "$cert" ]] || warn "DC4_CA_CERT is not set; set it in .devcontainer/local.env (see local.env.example)."
[[ -r "$cert" ]] || warn "cannot read $cert."
[[ $EUID -eq 0 ]] || warn "must run as root, so the certificate was not installed."
if grep -q 'PRIVATE KEY' "$cert"; then
  warn "$cert contains a private key: skipped. Only the public certificate belongs here."
fi
[[ "$(grep -c 'BEGIN CERTIFICATE' "$cert")" -eq 1 ]] || warn "$cert must hold exactly one PEM certificate: skipped."
subject="$(openssl x509 -in "$cert" -noout -subject 2>/dev/null)" || warn "$cert is not a valid certificate: skipped."
openssl x509 -in "$cert" -noout -checkend 0 >/dev/null || echo "setup-ca: WARNING: $cert has expired." >&2

if cp "$cert" "$dest" && chmod 644 "$dest" && update-ca-certificates >/dev/null 2>&1; then
  echo "setup-ca: trusting $subject"
else
  warn "could not update the system certificates."
fi
