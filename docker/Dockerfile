FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

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

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy the project
COPY . /nanochat

# Set up uv and install project deps
RUN . $HOME/.local/bin/env && \
    uv venv --python 3.11 && \
    uv sync --extra gpu && \
    echo "Verifying kernels package installation..." && \
    /nanochat/.venv/bin/python -c "from kernels import get_kernel, has_kernel; print('✓ kernels package installed successfully')"

# Make train/eval scripts executable
RUN chmod +x /nanochat/scripts/*.py /nanochat/runs/*.sh

# Create python symlinks
RUN ln -s /nanochat/.venv/bin/python3 /usr/local/bin/python && \
    ln -s /nanochat/.venv/bin/python3 /usr/local/bin/python3 && \
    ln -s /nanochat/.venv/bin/pip3 /usr/local/bin/pip

# Set PATH to include venv and uv
ENV PATH="/nanochat/.venv/bin:/root/.local/bin:$PATH"

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["bash"]