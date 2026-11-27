# CurlingCNN: CNN based curling AI

## Overview
CurlingCNN is a CNN based curling AI that works on DigitalCurling4.
- [Website](http://minerva.cs.uec.ac.jp/cgi-bin/curling/wiki.cgi?page=FrontPage)
- [GitHub](https://github.com/digitalcurling/DigitalCurling)

The CNN takes an image of sheet as input, and predicts the winning probability and the next shot.
The CNN is trained with the game statistics from World Curling Championships and Olympics.
This program is inspired by AlphaGo Zero and dlshogi.

## Running the client

The recommended way to run the client is to build the Docker image and start one container per team.

#### Requirements
- Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- An NVIDIA GPU with a driver that supports CUDA 13.4 or later (`nvidia-smi` shows "CUDA Version" 13.4 or higher)

Without a GPU the client runs on the CPU (it logs `CUDA is not available.`), but it searches far fewer
shots and plays much weaker.

#### 1. Download
- **Source:** the "Source code" archive of the version on the [Releases](https://github.com/pon-pokapoka/DigitalCurling4-CNN/releases) page,
  or `git clone --branch <tag> https://github.com/pon-pokapoka/DigitalCurling4-CNN.git`.
- **Model:** `traced.pt` and `win_table.csv`, attached to the same Release. They are not in the repository.

Put the model files into `src/engine/model/`. The source archive doesn't contain that directory, so create it:

```bash
cd DigitalCurling4-CNN
mkdir -p src/engine/model
cp /path/to/traced.pt /path/to/win_table.csv src/engine/model/
```

#### 2. Build
```bash
docker build -t cnn-client .
docker run --rm cnn-client --version    # check the build
```

`--version` prints the client name, its version and its build ID, e.g. `CNNClient 26.11.0 (build v26.11.0)`.
A build from a source download shows `build unknown`, because the download has no git history; the version
still tells which release it is. See [Versions](#versions).

#### 3. Run
Start one container per team. The client connects to the server, plays its team's side of one match, and
exits when the match is over.

```bash
docker run --rm --gpus all cnn-client \
    --host http://<server>:<port> --id <match-id> --team <0 or 1> \
    --auth-id <id> --auth-pw '<password>'
```

- Keep the password in single quotes: the shell would otherwise split a password containing characters such as `;`, `&` or `|`.

#### Options as environment variables
Every option can also be given as an environment variable (see [Options](#options)). Passing the password this
way keeps it out of the shell history and the process's command line:

```bash
read -rsp 'Password: ' DC_AUTH_PW && export DC_AUTH_PW
docker run --rm --gpus all \
    -e DC_HOST=http://<server>:<port> -e DC_MATCH_ID=<match-id> -e DC_TEAM=<0 or 1> \
    -e DC_AUTH_ID=<id> -e DC_AUTH_PW \
    cnn-client
```

`-e DC_AUTH_PW` without a value passes the variable from your shell.

#### TLS server with a private CA
The image trusts the public certificate authorities. For a server whose certificate is signed by a private CA,
mount the CA's public certificate and point `SSL_CERT_FILE` at it:

```bash
docker run --rm --gpus all \
    -v /path/to/ca.crt:/etc/dc4/ca.crt:ro -e SSL_CERT_FILE=/etc/dc4/ca.crt \
    cnn-client --host https://<server> --id <match-id> --team <0 or 1> \
    --auth-id <id> --auth-pw '<password>'
```

#### Troubleshooting
| Message | Cause and fix |
|---|---|
| `[Error] Failed to create HTTP client for host: ...` | The client got no HTTP response from the server: wrong host or port, a host name the container can't resolve (add `--add-host <name>:<IP>`), or a TLS certificate it doesn't trust (see [TLS server with a private CA](#tls-server-with-a-private-ca)). Check the server from the host with `curl https://<server>/`. |
| `CUDA is not available.` | The container sees no GPU: check `--gpus all`, the NVIDIA Container Toolkit, and the driver's CUDA version. |
| `[Error] Host and ID must be specified.` | `--host` and `--id` (or `DC_HOST` and `DC_MATCH_ID`) are required. |

## Options

#### Flags
| Argument | Description |
|---|---|
| `--help`, `-h` | Print the help message. |
| `--version`, `-v` | Print the version and build ID. |
| `--debug`, `-d` | Start in debug mode. |
| `--console`, `-c` | Read the host and match ID from standard input when they are not given. |

#### Options
| Argument | Environment variable | Description | Default |
|---|---|---|---|
| `--host` | `DC_HOST` | Server URL. | none |
| `--id` | `DC_MATCH_ID` | Match ID. | none |
| `--team` | `DC_TEAM` | Team to play (0 or 1). | 0 |
| `--auth-id` | `DC_AUTH_ID` | Basic authentication ID. | `user` |
| `--auth-pw` | `DC_AUTH_PW` | Basic authentication password. | `password` |
| `--plugin-dir` | | Directory to load plugins from. | `./plugins` |
| | `DC_CLIENT_THREADS` | Number of search threads. | all CPUs of the host |
| | `DC_TEAM_NAME_FULL` | `1`: add the build ID and the model and win table hashes to the team name sent to the server. For internal servers. | unset (short name) |

`--host` and `--id` are required to start the client. Every option can also be set with its environment
variable; when both are given, the argument wins. A password passed in `DC_AUTH_PW` doesn't appear in the
process's arguments.

## Building from source

#### Tools
- A C++17 compiler (GCC, Clang, MSVC, etc.)
- Git
- [CMake](https://cmake.org/) 3.26 or later
- [OpenSSL](https://www.openssl.org/), for HTTPS

#### Libraries
- LibTorch
- CUDA
- cuDNN

The container images provide these (see [Docker and dev container](#docker-and-dev-container)).
CMake fetches the following libraries itself:
- [DigitalCurling](https://github.com/digitalcurling/DigitalCurling) 4.0 or later
- [CLI11](https://github.com/CLIUtils/CLI11)
- [httplib](https://github.com/yhirose/cpp-httplib)

> [!Important]
> HTTPS needs OpenSSL installed on the system.

#### Build
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## Docker and dev container

GPU use requires the NVIDIA driver and the NVIDIA Container Toolkit on the host.

#### Development (dev container, CLI)
[`.devcontainer/`](.devcontainer/) defines the development image (CUDA devel + libtorch + gdb/clangd/ccache).
Start it with the [devcontainer CLI](https://github.com/devcontainers/cli) (`npm install -g @devcontainers/cli`):

```bash
.devcontainer/dev.sh up       # build the image and start the container (first run also configures CMake)
.devcontainer/dev.sh build    # cmake --build build inside the container
.devcontainer/dev.sh shell    # interactive shell inside the container
.devcontainer/dev.sh down     # stop and remove the container

# without the wrapper
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash
```

The container needs no SSH key, GitHub account or Claude. Personal extras are off by default;
turn them on in `.devcontainer/local.env` (not committed; copy
[`local.env.example`](.devcontainer/local.env.example)), which `dev.sh` reads:

| Option | Effect | Apply with |
|---|---|---|
| `DC4_SSH_AGENT=1` | Forward the host ssh-agent (git over SSH, commit signing) | `dev.sh down && dev.sh up` |
| `DC4_CLAUDE_CODE=1` | Install Claude Code in the image; keep its login in volume `dc4-cnn-claude` | `dev.sh rebuild` |
| `DC4_GITHUB=1` | Keep the GitHub CLI login in volume `dc4-cnn-gh` | `dev.sh down && dev.sh up` |
| `DC4_GIT_CONFIG=1` | Keep your git config (identity, signing) in volume `dc4-cnn-git` | `dev.sh down && dev.sh up` |
| `DC4_CA_CERT=<host path>` | Trust a private CA, for a TLS server that needs it: mounts that public certificate and runs [`setup-ca.sh`](.devcontainer/setup-ca.sh) on start | `dev.sh down && dev.sh up` |

Only the ccache volume is always created. To sign commits with your GitHub-registered SSH
signing key, set `DC4_SSH_AGENT=1` and `DC4_GIT_CONFIG=1` and run
`.devcontainer/setup-signing.sh` once inside the container.

#### Runtime image
The root [`Dockerfile`](Dockerfile) builds the client into an image of NVIDIA's `nvidia/cuda:*-cudnn-runtime`
(CUDA libraries, NCCL, cuDNN), libtorch's own shared libraries, the client binary and the model. It runs as a
non-root user (UID 10001) and works with a read-only root filesystem. `ENTRYPOINT` is the client itself and
there is no default command. How to build and run it is in [Running the client](#running-the-client).

The model files may be symbolic links, e.g. `traced.pt -> gen0-588f425c.pt` (see [Versions](#versions) for the
naming); the build copies their targets. From a git checkout, pass the build ID so the image identifies its code:

```bash
docker build --build-arg BUILD_ID="$(git describe --tags --dirty --always)" -t cnn-client:latest .
```

**Many builds, shared layers.** Everything but the client binary and the model is the same for every commit: the
`runtime-base` stage (the CUDA image, libtorch's libraries, Ubuntu updates, the user), about 3.3 GB compressed.
Build it once, push it, and build clients on it by digest; each client commit then adds a few MB, and a model
image made from it (a different `traced.pt` on top) a few MB more:

```bash
docker build --target runtime-base -t registry.example/cnn-runtime:torch2.15.0-cu134-2026-11 .
docker push registry.example/cnn-runtime:torch2.15.0-cu134-2026-11   # prints the digest
docker build --build-arg RUNTIME_BASE=registry.example/cnn-runtime@sha256:<digest> \
    --build-arg BUILD_ID="$(git describe --tags --dirty --always)" -t cnn-client:latest .
```

Without `RUNTIME_BASE` the build makes `runtime-base` itself; the image is the same, but its layers are shared
with other builds only while Docker's cache holds them.

**Security updates.** The base images are pinned by digest and updated by Renovate, but NVIDIA doesn't rebuild
them for Ubuntu fixes, so every stage runs `apt-get upgrade` (in `runtime-base` with NVIDIA's apt repository set
aside, so CUDA, cuDNN and NCCL stay at the pinned image's versions). Docker caches that step; to pick up Ubuntu
fixes released since the last build, rebuild `runtime-base` without the cache, push it, and rebuild the clients on
it (a few MB each; the build stage isn't shipped):

```bash
docker build --no-cache-filter runtime-base --target runtime-base -t registry.example/cnn-runtime:torch2.15.0-cu134-2026-12 .
```

For the dev container, `.devcontainer/dev.sh rebuild` does the same.

#### Kubernetes
[`deploy/`](deploy/README.md) has a Kustomize base that runs one client per team as a Job, with options as
environment variables and credentials from a Secret. Push the image to a registry your cluster can pull from and
point the overlay's `images:` at it.

For a TLS server with a private CA, add the certificate to the client container in the overlay:

```yaml
containers:
  - name: client
    env:
      - name: SSL_CERT_FILE
        value: /etc/dc4/ca.crt
    volumeMounts:
      - name: server-ca
        mountPath: /etc/dc4
        readOnly: true
volumes:
  - name: server-ca
    configMap:
      name: server-ca   # holds the public certificate as key ca.crt
```

## Versions

A client is its code, its model (`traced.pt`) and its win table (`win_table.csv`); each is identified separately.

| Part | Identified by | Example |
|---|---|---|
| Code, released | Calendar version `YY.MM.N` in `client_config.cmake` and the git tag `vYY.MM.N` | `v26.11.0` |
| Code, any build | Build ID: `git describe --tags --dirty --always`, with the `g` before the commit hash removed | `v2026.07-67-a604b03` |
| Model | SHA-256 of the file; model files are named `<name>-<first 8 hex>.pt` | `gen0-588f425c` |
| Win table | SHA-256 of the file; named `win_table-<first 8 hex>.csv` | `648e2603` |

- The year has two digits because the version parts are stored in one byte each (`digitalcurling::Version`).
- The version in `client_config.cmake` changes only at a release. Between releases, the build ID identifies the code. A build with uncommitted changes ends in `-dirty`.
- `--version` prints the version and the build ID. At start-up the client logs the model and win table files it loaded and their SHA-256.
- The team name sent to the server is the client name (`CNNClient`), as tournaments expect. With `DC_TEAM_NAME_FULL=1` it is
  `<name>-<build ID>-<model hash>-<table hash>`, e.g. `CNNClient-v2026.07-67-a604b03-588f425c-648e2603`, so an internal server's match records show exactly which client played.

## Developing a thinking engine

Thinking engines live in [src/engine/](src/engine/).

1. Create the engine class. Derive it from the interface for the rules it supports; a class may derive from
   several to support several rules.
   - `IStandardThinkingEngine` (standard curling)
   - `IMixedThinkingEngine` (mixed curling)
   - `IMixedDoublesThinkingEngine` (mixed doubles curling)

1. Edit the functions in `src/client_setup.cpp` so that they return your engine class. Change the rest of the
   code as needed.

1. Set the client information in `client_config.cmake` at the repository root.

## Model
The network takes the sheet, score difference, and shot number as inputs.
Policy network selects the next shot (speed, angle, and rotation).
Value network predict the score of the end, which is converted to the winning rate with the number of ends to be played and the score difference.

## Training
Game statistics extracted from shot by shot available on [curlit](https://curlit.com/results) are used to train the model. Data extraction is described [here](https://www.jordanmyslik.com/portfolio/curling-analytics/).

## Strategy
CurlingCNN estimates shot candidates (speed, angle, and rotation) from the current state.
Then, it adopts a shot where the model returns the highest winning probability after simulation of shot candidates.
This strategy is inspired by AlphaGo Zero and dlshogi.

## Author
Takehiro Yoshioka
- [@pon_pokapoka](https://x.com/pon_pokapoka)

## License

[MIT](./LICENSE)
