#!/bin/bash
# Docker entrypoint script for NanoChat curriculum training

set -e

echo "========================================"
echo "NanoChat Curriculum Training Container"
echo "========================================"
echo ""

# Activate Python virtual environment
if [ -f "/workspace/.venv/bin/activate" ]; then
    echo "Activating Python virtual environment..."
    source /workspace/.venv/bin/activate
else
    echo "WARNING: Virtual environment not found at /workspace/.venv"
fi

# Display environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "GPU 0: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Execute the provided command
exec "$@"
