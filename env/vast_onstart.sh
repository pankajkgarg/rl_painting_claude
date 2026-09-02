#!/bin/bash
# vast.ai onstart: install deps for Qwen3.5-9B GRPO + reward_v1 scorers; touch /root/READY when done.
exec > /root/onstart.log 2>&1
set -x
apt-get update -qq && apt-get install -y -qq curl >/dev/null
pip install -q --upgrade unsloth unsloth_zoo trl playwright open_clip_torch transformers sentencepiece
python -m playwright install --with-deps chromium
mkdir -p /root/env /root/out
curl -sL -o /root/env/HPS_v2.1_compressed.pt https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
python - <<'PY'
from transformers import AutoModel, AutoProcessor
AutoModel.from_pretrained("google/siglip2-base-patch16-224"); AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Qwen3.5-9B")
PY
pip list 2>/dev/null | grep -iE '^(torch|transformers|trl|unsloth|unsloth_zoo|vllm|playwright|open_clip_torch) '
nvidia-smi --query-gpu=name,memory.total --format=csv
touch /root/READY
