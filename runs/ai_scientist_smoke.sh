#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-opencode/space-bunny-free}"
docker compose --profile ai-scientist run --rm ai-scientist \
  python launch_scientist_bfts.py --load-code --max-nodes 1 --allow-provider-calls --model "$MODEL"
