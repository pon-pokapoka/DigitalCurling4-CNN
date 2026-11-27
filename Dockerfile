# syntax=docker/dockerfile:1
#
# Runtime image for the CNN DigitalCurling client (Kubernetes-ready).
#
#   docker build -t cnn-client:latest .
#   docker run --rm --gpus all -e DC_AUTH_ID=user -e DC_AUTH_PW cnn-client:latest \
#       --host http://server:5000 --id <match> --team 0
#
# For a server with a private CA, mount the CA certificate and set SSL_CERT_FILE:
#   docker run ... -v "$PWD/ca.crt:/etc/dc4/ca.crt:ro" -e SSL_CERT_FILE=/etc/dc4/ca.crt ...
#
# Stages:
#   toolchain    : CUDA devel + compilers + libtorch (cached; changes only when versions change)
#   build        : compiles the client and stages the app: binary + default model
#   runtime-libs : libtorch's own shared libraries, without the CUDA, cuDNN and NCCL
#                  libraries that the cuDNN runtime image already has
#   runtime-base : NVIDIA's cuDNN runtime image + those libraries + Ubuntu updates + user;
#                  everything that is the same for every client commit and model
#   runtime      : runtime-base (or RUNTIME_BASE) + the app, non-root
#
# The development image lives in .devcontainer/Dockerfile.
#
# Sharing layers between builds: layers are ordered from what changes least to what
# changes most (CUDA + cuDNN, libtorch, Ubuntu updates, the app), so a new client
# commit adds only the app layer (a few MB) when the layers below it are unchanged.
# They are unchanged only if runtime-base is: build it once, push it, and pass it
# pinned by digest, so builds don't depend on the layer cache:
#   docker build --target runtime-base -t registry.example/cnn-runtime:<version> . && docker push ...
#   docker build --build-arg RUNTIME_BASE=registry.example/cnn-runtime@sha256:<digest> ...
# Without RUNTIME_BASE the image is self-contained and builds runtime-base here.
#
# Security updates: runtime-base runs `apt-get upgrade` for Ubuntu's packages, because
# NVIDIA does not rebuild its images for Ubuntu fixes. Rebuild runtime-base for new
# fixes, without the layer cache, and rebuild the clients on it:
#   docker build --no-cache-filter runtime-base --target runtime-base ...
# NVIDIA's apt repository is set aside for that upgrade, so CUDA, cuDNN and NCCL stay
# at the pinned image's versions; they change only with the image digests below.
#
# Base images are pinned by digest and updated by Renovate. Keep the toolchain and
# runtime images on the same CUDA and cuDNN version: the client is compiled against
# the devel image and runs on the runtime image's libraries.

ARG RUNTIME_BASE=runtime-base

# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.4.3-cudnn-devel-ubuntu26.04@sha256:a34cd707296198f7bf581ab8da0a977e2d6cea912ba081cf872c6742e3ade503 AS toolchain

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        g++ \
        git \
        libssl-dev \
        ninja-build \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# renovate: datasource=pypi depName=torch
ARG TORCH_VERSION=2.15.0
ARG TORCH_CUDA=134
# SHA256 of the archive below: change it together with the two versions above,
# including when overriding them with --build-arg.
ARG LIBTORCH_SHA256=786f18f4af224e1caa2f649c8c0db6c3cc3dda98f47e6d8cd863a3c3b86859f5
RUN curl -fsSL --retry 3 -o /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cu${TORCH_CUDA}/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu${TORCH_CUDA}.zip" \
    && echo "${LIBTORCH_SHA256}  /tmp/libtorch.zip" > /tmp/libtorch.sha256 \
    && sha256sum -c /tmp/libtorch.sha256 \
    && unzip -q /tmp/libtorch.zip -d /usr/local \
    && rm /tmp/libtorch.zip /tmp/libtorch.sha256

# ---------------------------------------------------------------------------
FROM toolchain AS build

WORKDIR /src
COPY CMakeLists.txt client_config.cmake ./
COPY include/ include/
COPY src/ src/

# The build context has no .git, so pass the build ID in:
#   --build-arg BUILD_ID=$(git describe --tags --dirty --always)
# Without it the build ID is "unknown".
ARG BUILD_ID=""
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DDIGITALCURLING_CLIENT_BUILD_ID="${BUILD_ID}" \
    && cmake --build build --parallel

# Stage the app. The engine resolves its model as <exe>/../../../src/engine/model,
# so the binary must live at <root>/build/bin/ and the model at <root>/src/engine/model/.
RUN set -eux; \
    mkdir -p /out/app/build/bin /out/app/src/engine/model; \
    cp build/bin/CNNClient_v* /out/app/build/bin/cnn-client; \
    cp src/engine/model/traced.pt src/engine/model/win_table.csv /out/app/src/engine/model/

# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.4.3-cudnn-runtime-ubuntu26.04@sha256:fa18099d892a6d36e63208084216e958790145bbcfe720f58fd874544133223f AS runtime-libs

# libtorch's shared libraries, except test/python-only ones and those the runtime
# image already has under the same name (cuBLAS, cuFFT, cuSPARSE, cuSOLVER, NCCL,
# NVRTC, cuDNN, ...): those are used from NVIDIA's layers, stored once. Everything
# else is copied, not chosen by ldd, because libtorch dlopen()s some libraries lazily.
COPY --from=toolchain /usr/local/libtorch/lib/ /tmp/libtorch/
# ldconfig first: the image's linker cache doesn't list all of its libraries (cuDNN).
RUN set -eux; \
    ldconfig; \
    ldconfig -p > /tmp/ld-cache.txt; \
    mkdir -p /out/usr/local/libtorch/lib; \
    for p in /tmp/libtorch/*.so*; do \
        n="${p##*/}"; \
        case "$n" in \
            libtorch_python.so|*_test.so|libbackend_with_compiler.so|libnnapi_backend.so|libaoti_custom_ops.so) continue ;; \
        esac; \
        if grep -q "^[[:space:]]*${n} " /tmp/ld-cache.txt; then continue; fi; \
        cp -a "$p" /out/usr/local/libtorch/lib/; \
    done

# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.4.3-cudnn-runtime-ubuntu26.04@sha256:fa18099d892a6d36e63208084216e958790145bbcfe720f58fd874544133223f AS runtime-base

COPY --link --from=runtime-libs /out/ /

# Ubuntu's fixes (glibc, zlib, OpenSSL, ...) on top of libtorch, so a new upgrade
# leaves the layers below shared. NVIDIA's repository is set aside meanwhile.
# Fixed numeric UID so Kubernetes `runAsNonRoot: true` can verify it.
ARG APP_UID=10001
RUN set -eux; \
    mkdir /tmp/nvidia-apt; \
    mv /etc/apt/sources.list.d/cuda* /tmp/nvidia-apt/; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libssl3t64; \
    rm -rf /var/lib/apt/lists/*; \
    mv /tmp/nvidia-apt/* /etc/apt/sources.list.d/; \
    rmdir /tmp/nvidia-apt; \
    useradd --system --uid ${APP_UID} --user-group --no-create-home --home-dir /app app

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/libtorch/lib

# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE} AS runtime

COPY --link --from=build /out/ /

ARG APP_UID=10001
USER ${APP_UID}:${APP_UID}
WORKDIR /app

LABEL org.opencontainers.image.title="CNNClient" \
      org.opencontainers.image.source="https://github.com/pon-pokapoka/DigitalCurling4-CNN" \
      org.opencontainers.image.licenses="MIT"

# Exec form: the client is PID 1 and receives SIGTERM directly from the kubelet.
# Options come from DC_* environment variables (or arguments). No default CMD:
# with one, a Pod that sets only env would run that CMD instead of the client.
ENTRYPOINT ["/app/build/bin/cnn-client"]
