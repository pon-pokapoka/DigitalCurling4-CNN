# syntax=docker/dockerfile:1
#
# Development image for the dev container (.devcontainer/devcontainer.json).
# The source tree is bind-mounted at runtime; nothing from the repo is copied in.

ARG CUDA_VERSION=13.4.3
ARG UBUNTU_VERSION=26.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # build
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
        # dev tools
        bash-completion \
        ccache \
        clang-format \
        clangd \
        curl \
        gdb \
        jq \
        less \
        openssh-client \
        sudo \
        valgrind \
        vim \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.15.0
ARG TORCH_CUDA=134
RUN wget -q -O /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cu${TORCH_CUDA}/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu${TORCH_CUDA}.zip" \
    && unzip -q /tmp/libtorch.zip -d /usr/local \
    && rm /tmp/libtorch.zip

# Ubuntu (24.04+) images ship a non-root `ubuntu` user (UID 1000).
# The devcontainer CLI remaps its UID/GID to the host user's (updateRemoteUserUID),
# so bind-mounted files keep the right ownership.
ARG USERNAME=ubuntu
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    # pre-create the ccache dir so the named volume inherits user ownership
    && mkdir -p /home/${USERNAME}/.cache/ccache \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cache

ENV CCACHE_DIR=/home/${USERNAME}/.cache/ccache \
    CMAKE_GENERATOR=Ninja \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Claude Code (native installer). Copied to /usr/local/bin so any user can run it;
# rebuild the image to update (auto-updater disabled).
# Config/login lives in CLAUDE_CONFIG_DIR, backed by a named volume (see devcontainer.json).
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && cp -L /root/.local/bin/claude /usr/local/bin/claude \
    && rm -rf /root/.local/bin/claude /root/.local/share/claude \
    && mkdir -p /opt/claude-config \
    && chown ${USERNAME}:${USERNAME} /opt/claude-config
ENV CLAUDE_CONFIG_DIR=/opt/claude-config \
    DISABLE_AUTOUPDATER=1
