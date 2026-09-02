#!/usr/bin/env bash
# Paid fallback: rent a cheap RTX 3090 on vast.ai (~$0.11-0.18/hr), run train.py with vLLM fast_inference.
# Requires: vastai CLI logged in + balance (>= $3 is plenty for a 100-step run).
# Usage: env/vast_launch.sh            # picks cheapest verified 3090, prints instance id
set -euo pipefail
cd "$(dirname "$0")/.."
OFFER=$(vastai search offers 'gpu_name=RTX_3090 num_gpus=1 rentable=true verified=true disk_space>=50 inet_down>300 reliability>0.98' -o 'dph' --raw | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['id'])")
echo "offer $OFFER"
vastai create instance "$OFFER" --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 50 --ssh --direct \
  --onstart-cmd 'pip install -q unsloth trl vllm playwright open_clip_torch && python -m playwright install --with-deps chromium'
echo "then: vastai scp <id> env/ /root/env ; ssh in; cd /root/env && RLPAINT_VLLM=1 nohup python train.py --steps 100 --gens 8 > train.log &"
echo "ALWAYS: vastai destroy instance <id> when done."
