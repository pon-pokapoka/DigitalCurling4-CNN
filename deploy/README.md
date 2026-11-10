# Kubernetes manifests

`deploy/base/` is a generic [Kustomize](https://kustomize.io/) base that runs one CNN client:
one Job per team per match. It contains no hosts, credentials, or cluster-specific settings.
Keep those in a private overlay repo that references this base:

```yaml
resources:
  - https://github.com/pon-pokapoka/DigitalCurling4-CNN//deploy/base?ref=<tag-or-commit>
```

## What the base defines

| Item | Default | Override in the overlay |
|---|---|---|
| Image | placeholder `cnn-client` | `images:` with your registry and tag |
| Server URL | ConfigMap `cnn-client`, key `host` | `configMapGenerator` with `behavior: replace` |
| Match ID | env `DC_MATCH_ID` (empty) | patch the env entry (required) |
| Team | env `DC_TEAM` = `0` | patch the env entry |
| Credentials | Secret `cnn-client-auth` (keys `auth-id`, `auth-pw`) | create in the cluster; point each team at its own Secret |
| Team name | client name only (`CNNClient`) | env `DC_TEAM_NAME_FULL: "1"` adds the build ID and model hashes; for internal servers |
| Search threads | `DC_CLIENT_THREADS` = CPU limit; with no limit, all node CPUs | set `resources.limits.cpu` to cap it |
| Resources | request 1 CPU / 2 GiB; no CPU limit; 6 GiB and 1 `nvidia.com/gpu` limit | patch `resources` |
| Timeouts | 3 h deadline, cleaned up 1 h after finishing | `activeDeadlineSeconds`, `ttlSecondsAfterFinished` |

The pod runs as UID 10001 with a read-only root filesystem, no capabilities, and no
service account token.

## Credentials

The Secret is never part of the manifests. Create one per team in the cluster:

```bash
kubectl create secret generic cnn-client-auth-team0 \
    --from-literal=auth-id=<id> --from-literal=auth-pw=<password>
```

The client reads its options from environment variables (`DC_HOST`, `DC_MATCH_ID`,
`DC_TEAM`, `DC_AUTH_ID`, `DC_AUTH_PW`), so the Job passes no arguments and the password
never appears in a process's command line. This needs an image built with that support
(DigitalCurling4-CNN #30 or later).

## Example overlay (one team)

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - https://github.com/pon-pokapoka/DigitalCurling4-CNN//deploy/base?ref=<tag-or-commit>
nameSuffix: -team0
images:
  - name: cnn-client
    newName: registry.example/cnn-client
    newTag: v0.1.0
configMapGenerator:
  - name: cnn-client
    behavior: replace
    literals:
      - host=https://server.example
patches:
  - target: {kind: Job}
    patch: |-
      apiVersion: batch/v1
      kind: Job
      metadata: {name: cnn-client}
      spec:
        template:
          spec:
            containers:
              - name: client
                env:
                  - {name: DC_MATCH_ID, value: "<match-id>"}
                  - {name: DC_TEAM, value: "0"}
                  - name: DC_AUTH_ID
                    valueFrom: {secretKeyRef: {name: cnn-client-auth-team0, key: auth-id}}
                  - name: DC_AUTH_PW
                    valueFrom: {secretKeyRef: {name: cnn-client-auth-team0, key: auth-pw}}
```

Env entries are merged by `name`, so a patch only needs the entries it changes.
For team 1, use `nameSuffix: -team1`, `DC_TEAM: "1"`, and the team-1 Secret.

## GPU scheduling

Each client requests one `nvidia.com/gpu`. To run both teams of a match (or several matches)
on one physical GPU, enable time-slicing in the NVIDIA device plugin. Depending on the cluster,
the overlay may also need `runtimeClassName: nvidia`, node selectors, or tolerations for GPU nodes.

For a TLS server with a private CA, mount the CA certificate and set `SSL_CERT_FILE`;
see [the main README](../README.md#runtime-image-kubernetes).
