# syntax=docker/dockerfile:1
#
# Runtime image for the CNN DigitalCurling client (Kubernetes-ready).
#
#   docker build -t cnn-client:latest .
#   docker run --rm --gpus all cnn-client:latest --host http://server:5000 --id <match> --team 0 \
#       --auth-id user --auth-password pass
#
# Stages:
#   toolchain : CUDA devel + compilers + libtorch (cached; changes only when versions change)
#   build     : compiles the client and stages a minimal /out tree
#   runtime   : nvidia/cuda *base* image + libtorch shared libs + binary + model, non-root
#
# The development image lives in .devcontainer/Dockerfile.

ARG CUDA_VERSION=13.4.3
ARG UBUNTU_VERSION=26.04

# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS toolchain

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        libboost-dev \
        libboost-date-time-dev \
        libboost-regex-dev \
        libicu-dev \
        libssl-dev \
        ninja-build \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.15.0
ARG TORCH_CUDA=134
RUN wget -q -O /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cu${TORCH_CUDA}/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu${TORCH_CUDA}.zip" \
    && unzip -q /tmp/libtorch.zip -d /usr/local \
    && rm /tmp/libtorch.zip

# ---------------------------------------------------------------------------
FROM toolchain AS build

WORKDIR /src
COPY CMakeLists.txt client_config.cmake ./
COPY include/ include/
COPY src/ src/

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
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
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
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
# Pass connection options as container `args:` in the Pod spec.
ENTRYPOINT ["/app/build/bin/cnn-client"]
CMD ["--help"]
