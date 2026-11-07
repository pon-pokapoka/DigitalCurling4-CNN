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
        # dev tools
        bash-completion \
        ccache \
        clang-format \
        clangd \
        curl \
        gdb \
        gh \
        jq \
        less \
        openssh-client \
        sudo \
        valgrind \
        vim \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.15.0
ARG TORCH_CUDA=134
# SHA256 of the archive below: change it together with the two versions above.
# devcontainer.json overrides all three, so keep them in sync there.
ARG LIBTORCH_SHA256=786f18f4af224e1caa2f649c8c0db6c3cc3dda98f47e6d8cd863a3c3b86859f5
RUN curl -fsSL --retry 3 -o /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cu${TORCH_CUDA}/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu${TORCH_CUDA}.zip" \
    && echo "${LIBTORCH_SHA256}  /tmp/libtorch.zip" > /tmp/libtorch.sha256 \
    && sha256sum -c /tmp/libtorch.sha256 \
    && unzip -q /tmp/libtorch.zip -d /usr/local \
    && rm /tmp/libtorch.zip /tmp/libtorch.sha256

# Ubuntu (24.04+) images ship a non-root `ubuntu` user (UID 1000).
# The devcontainer CLI remaps its UID/GID to the host user's (updateRemoteUserUID),
# so bind-mounted files keep the right ownership.
# A login name, not a secret (hadolint DL3064 matches on "USER" in the name).
# hadolint ignore=DL3064
ARG USERNAME=ubuntu
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    # pre-create the ccache dir so the named volume inherits user ownership
    && mkdir -p /home/${USERNAME}/.cache/ccache \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cache \
    # personal git config (GIT_CONFIG_GLOBAL), backed by a named volume when
    # DC4_GIT_CONFIG=1 (see dev.sh)
    && mkdir -p /opt/git-config \
    && chown ${USERNAME}:${USERNAME} /opt/git-config

ENV CCACHE_DIR=/home/${USERNAME}/.cache/ccache \
    CMAKE_GENERATOR=Ninja \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Claude Code (native installer), only with INSTALL_CLAUDE_CODE=1 (off by default;
# see .devcontainer/local.env.example). Copied to /usr/local/bin so any user can
# run it; rebuild the image to update (auto-updater disabled).
# Config/login lives in CLAUDE_CONFIG_DIR, backed by a named volume (see dev.sh).
# Fail the build if the download fails, not only if bash does (hadolint DL4006).
ARG INSTALL_CLAUDE_CODE=0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN if [ "${INSTALL_CLAUDE_CODE}" = "1" ]; then \
        curl -fsSL https://claude.ai/install.sh | bash \
        && cp -L /root/.local/bin/claude /usr/local/bin/claude \
        && rm -rf /root/.local/bin/claude /root/.local/share/claude; \
    fi \
    && mkdir -p /opt/claude-config \
    && chown ${USERNAME}:${USERNAME} /opt/claude-config
ENV CLAUDE_CONFIG_DIR=/opt/claude-config \
    DISABLE_AUTOUPDATER=1
