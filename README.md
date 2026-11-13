# CurlingCNN: CNN based curling AI

## Overview
CurlingCNN is a CNN based curling AI that works on DigitalCurling4.
- [Website](http://minerva.cs.uec.ac.jp/cgi-bin/curling/wiki.cgi?page=FrontPage)
- [GitHub](https://github.com/digitalcurling/DigitalCurling)

The CNN takes an image of sheet as input, and predicts the winning probability and the next shot.
The CNN is trained with the game statistics from World Curling Championships and Olympics.
This program is inspired by AlphaGo Zero and dlshogi.

## 必要要件

#### ツール
- C++17 対応のコンパイラ (GCC, Clang, MSVC, etc.)
- Git
- [CMake](https://cmake.org/) 3.26 以上
- [OpenSSL](https://www.openssl.org/) (HTTPS通信を行う場合)

#### 依存ライブラリ
- LibTorch
- CUDA
- CuDNN
These required libraries are provided by the container images (see [Docker / Dev Container](#docker--dev-container)).

以下のライブラリは CMake によって自動的に解決されます。
- [DigitalCurling](https://github.com/digitalcurling/DigitalCurling) 4.0 以上
- [CLI11](https://github.com/CLIUtils/CLI11)
- [httplib](https://github.com/yhirose/cpp-httplib)

> [!Important]
> HTTPS通信を行う場合、OpenSSLが必要です。
> OpenSSLはシステムにインストールされている必要があります。

## ビルド

1. ビルド用ディレクトリを作成し、ビルドを実行します。
   ```bash
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --config Release
   ```

## Docker / Dev Container

GPU use requires the NVIDIA driver and the NVIDIA Container Toolkit on the host.

#### Development (Dev Container, CLI)
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

#### Runtime image (Kubernetes)
The root [`Dockerfile`](Dockerfile) is a multi-stage build. The final image is based on `nvidia/cuda:*-base`
and contains only the binary, the libtorch shared libraries, and the model.
It runs as a non-root user (UID 10001), and `ENTRYPOINT` is the client itself, so pass connection options via the Pod's `args:`.
A Kustomize base that runs one client per team as a Job is in [`deploy/`](deploy/README.md).
The model files (`src/engine/model/traced.pt`, `win_table.csv`) must be present in the build context.

```bash
docker build --build-arg BUILD_ID="$(git describe --tags --dirty --always)" -t cnn-client:latest .
docker run --rm --gpus all -e DC_AUTH_ID=user -e DC_AUTH_PW cnn-client:latest \
    --host http://server:5000 --id <match-id> --team 0
```

`-e DC_AUTH_PW` without a value passes the variable from your shell, so the password isn't on the command line.

**Security updates.** The base images are pinned by digest and updated by Renovate, but NVIDIA doesn't rebuild
them for Ubuntu fixes, so every stage runs `apt-get upgrade`. Docker caches that step; to pick up Ubuntu fixes
released since the last build, rebuild the runtime stage without the cache (the build stage isn't shipped):

```bash
docker build --no-cache-filter runtime --build-arg BUILD_ID="$(git describe --tags --dirty --always)" -t cnn-client:latest .
```

For the dev container, `.devcontainer/dev.sh rebuild` does the same.

The image trusts the public CAs. For a TLS server with a private CA, mount the CA's public certificate
and point `SSL_CERT_FILE` at it. This works with a non-root user and a read-only root filesystem:

```bash
docker run --rm --gpus all -e DC_AUTH_ID=user -e DC_AUTH_PW \
    -v "$PWD/ca.crt:/etc/dc4/ca.crt:ro" -e SSL_CERT_FILE=/etc/dc4/ca.crt \
    cnn-client:latest \
    --host https://server --id <match-id> --team 0
```

```yaml
# Example Pod spec
containers:
  - name: client
    image: cnn-client:latest
    args: ["--host", "http://server:5000", "--id", "$(MATCH_ID)", "--team", "0"]
    # Credentials as environment variables from a Secret, never in args.
    envFrom:
      - secretRef:
          name: cnn-client-auth   # keys DC_AUTH_ID and DC_AUTH_PW
    securityContext:
      runAsNonRoot: true
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
    resources:
      limits:
        nvidia.com/gpu: 1
    # Only for a TLS server with a private CA:
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

## 使用方法

ビルド成果物を実行することで、思考エンジンクライアントが起動します。
以下の実行引数が使用可能です。

#### フラグ
| 引数 | 説明 |
|------|------|
| `--help`, `-h` | ヘルプメッセージを表示します。 |
| `--version`, `-v` | バージョン情報を表示します。 |
| `--debug`, `-d` | デバッグモードを有効にして起動します。 |
| `--console`, `-c` | コンソール入力を有効にして起動します。 |

#### オプション
| 引数 | 環境変数 | 説明 | デフォルト値 |
|------|----------|------|--------------|
| `--host` | `DC_HOST` | 接続先ホストを指定します。 | none |
| `--id` | `DC_MATCH_ID` | 接続先のゲームIDを指定します。 | none |
| `--team` | `DC_TEAM` | 接続するチームを指定します。(0または1) | 0 |
| `--auth-id` | `DC_AUTH_ID` | Basic認証のIDを指定します。 | `user` |
| `--auth-pw` | `DC_AUTH_PW` | Basic認証のパスワードを指定します。 | `password` |
| `--plugin-dir` | | プラグインのディレクトリを指定します。 | `./plugins` |
| | `DC_TEAM_NAME_FULL` | `1` で、サーバーに送るチーム名にビルドIDとモデル・勝率表のハッシュを付けます。内部サーバー向け。 | 未設定 (短い名前) |

オプションは全て任意オプションですが、`--host` および `--id` はクライアントの起動に必要です。  
各オプションは環境変数でも指定でき、両方ある場合は引数が優先されます。
パスワードは `DC_AUTH_PW` で渡すと、プロセスの引数に現れません。  
`--console` フラグ指定を指定した場合は、標準入力にて接続先情報を入力することができます。

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
- `--version` prints the build ID. At start-up the client logs the model and win table files it loaded and their SHA-256.
- The team name sent to the server is the client name (`CNNClient`), as tournaments expect. With `DC_TEAM_NAME_FULL=1` it is
  `<name>-<build ID>-<model hash>-<table hash>`, e.g. `CNNClient-v2026.07-67-a604b03-588f425c-648e2603`, so an internal server's match records show exactly which client played.

## 思考エンジンの開発方法

思考エンジンは、[src/engine/](src/engine/) ディレクトリ内に配置。

#### 開発手順

1. 思考エンジンのクラスを作成します。  
対応させるルールに応じて、以下のインターフェイスクラスを継承してください。
複数のクラスを継承し、複数のルールに対応する思考エンジンを作成することも可能です。
   - `IStandardThinkingEngine` (通常のカーリング用)
   - `IMixedThinkingEngine` (ミックスカーリング用)
   - `IMixedDoublesThinkingEngine` (ミックスダブルスカーリング用)

1. `src/client_setup.cpp` 内の関数を編集し、作成した思考エンジンのクラスを返すようにします。  
その他のコードは、必要に応じて編集してください。

1. ルートディレクトリの `client_config.cmake` でクライアント情報を設定します。

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

## ライセンス

[MIT](./LICENSE)
