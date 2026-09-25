# NanoChat Curriculum Training - Docker Image
# Base: PyTorch 2.9.1 with CUDA 12.8 support

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (for layer caching)
COPY pyproject.toml uv.lock /workspace/

# Install uv (fast Python package manager) and create venv in one layer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv venv && \
    . .venv/bin/activate && \
    uv sync --extra gpu

# Now copy the rest of the project files
COPY . /workspace/

# Add uv to PATH permanently
ENV PATH="/root/.local/bin:$PATH"

# Set environment variables
ENV NANOCHAT_BASE_DIR=/workspace/data
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

# Create data directories
RUN mkdir -p /workspace/data /workspace/checkpoints

# Make scripts executable
RUN chmod +x docker/entrypoint.sh docker/init_training.sh runs/*.sh

# Entrypoint
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["/bin/bash"]
