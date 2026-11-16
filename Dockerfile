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
#   toolchain : CUDA devel + compilers + libtorch (cached; changes only when versions change)
#   build     : compiles the client and stages a minimal /out tree
#   runtime   : nvidia/cuda *base* image + libtorch shared libs + binary + model, non-root
#
# The development image lives in .devcontainer/Dockerfile.
#
# Security updates: every stage runs `apt-get upgrade`, because NVIDIA does not
# rebuild its images for Ubuntu fixes. To pick up new fixes without a version
# change, rebuild the runtime stage without the layer cache:
#   docker build --no-cache-filter runtime ...
#
# Base images are pinned by digest and updated by Renovate. Keep the toolchain
# and runtime images on the same CUDA version: the runtime stage copies cuDNN and
# NVRTC libraries from the toolchain image.

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

# Stage the runtime tree.
# The engine resolves its model as <exe>/../../../src/engine/model, so the
# binary must live at <root>/build/bin/ and the model at <root>/src/engine/model/.
# libtorch dlopen()s cuDNN/NVRTC lazily, so copy every shared lib except
# test/python-only ones rather than relying on ldd.
RUN set -eux; \
    mkdir -p /out/app/build/bin /out/app/src/engine/model /out/usr/local/libtorch/lib; \
    cp build/bin/CNNClient_v* /out/app/build/bin/cnn-client; \
    cp src/engine/model/traced.pt src/engine/model/win_table.csv /out/app/src/engine/model/; \
    for p in /usr/local/libtorch/lib/*.so*; do \
        case "${p##*/}" in \
            libtorch_python.so|*_test.so|libbackend_with_compiler.so|libnnapi_backend.so|libaoti_custom_ops.so) ;; \
            *) cp -a "$p" /out/usr/local/libtorch/lib/ ;; \
        esac; \
    done; \
    # libtorch (>=2.10) bundles only the cuDNN dispatcher (libcudnn.so.9), not the
    # sub-libraries it dlopen()s (ops/cnn/engines...), nor NVRTC builtins used by the
    # TorchScript CUDA fuser. Take a consistent set from the devel image.
    cp -a /usr/lib/x86_64-linux-gnu/libcudnn*.so.9* /out/usr/local/libtorch/lib/; \
    cp -a /usr/local/cuda/lib64/libnvrtc-builtins.so* /out/usr/local/libtorch/lib/

# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.4.3-base-ubuntu26.04@sha256:e2eb43e25d404e2c48716f80afdb8989a84c4da588f5509330322842084d97b0 AS runtime

# The upgrade brings the base image's Ubuntu packages (glibc, zlib, ...) up to
# date; NVIDIA's images are often weeks behind. The client's CUDA libraries come
# from libtorch and the toolchain stage, so this doesn't change them.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/ /

# Fixed numeric UID so Kubernetes `runAsNonRoot: true` can verify it.
ARG APP_UID=10001
RUN useradd --system --uid ${APP_UID} --user-group --no-create-home --home-dir /app app
USER ${APP_UID}:${APP_UID}
WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/libtorch/lib

LABEL org.opencontainers.image.title="CNNClient" \
      org.opencontainers.image.source="https://github.com/pon-pokapoka/DigitalCurling4-CNN" \
      org.opencontainers.image.licenses="MIT"

# Exec form: the client is PID 1 and receives SIGTERM directly from the kubelet.
# Options come from DC_* environment variables (or arguments). No default CMD:
# with one, a Pod that sets only env would run that CMD instead of the client.
ENTRYPOINT ["/app/build/bin/cnn-client"]
