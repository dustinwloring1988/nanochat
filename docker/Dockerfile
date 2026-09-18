FROM nvidia/cuda:13.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /nanochat

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    ca-certificates \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv and deps in one layer so PATH persists
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    . $HOME/.local/bin/env && \
    uv --version && \
    # Force uv to use Python 3.11
    uv venv --python 3.11

# Copy the project
COPY . /nanochat

# Set up uv default-groups and install project deps
RUN . $HOME/.local/bin/env && uv sync --extra gpu

# Make train/eval scripts executable
RUN chmod +x /nanochat/scripts/*.py /nanochat/runs/*.sh

# Default command
CMD ["bash"]