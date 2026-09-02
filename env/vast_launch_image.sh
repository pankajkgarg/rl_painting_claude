#!/usr/bin/env bash
# Rent a 45GB+ GPU with the prebuilt image; ready in ~2-3 min instead of ~20.
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/pankajkgarg/rlpaint:latest}
OFFER=$(vastai search offers 'gpu_name in [L40,L40S,RTX_6000Ada,A6000,RTX_A6000,A40,A100_SXM4,A100_PCIE] num_gpus=1 rentable=true verified=true disk_space>=60 inet_down>300 reliability>0.97 cuda_max_good>=13.0' -o 'dph' --raw | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['id'])")
vastai create instance "$OFFER" --image "$IMAGE" --disk 60 --ssh --direct \
  --onstart-cmd 'python -c "from huggingface_hub import snapshot_download; snapshot_download(\"unsloth/Qwen3-4B-Instruct-2507\")" && touch /root/READY'
