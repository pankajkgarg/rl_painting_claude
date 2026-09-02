#!/usr/bin/env bash
# Rent a GPU with the prebuilt image (ghcr.io/pankajkgarg/rlpaint): training-ready in ~3 min + model download.
# Usage: GPU=24|48|80 MODEL=Qwen/Qwen3.5-9B env/vast_launch_image.sh
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-ghcr.io/pankajkgarg/rlpaint:latest}; GPU=${GPU:-24}; MODEL=${MODEL:-Qwen/Qwen3.5-4B}
case "$GPU" in
  80) NAMES='[A100_SXM4,A100_PCIE,H100_SXM,H100_PCIE,H100_NVL,H200]'; MINRAM=79000 ;;
  48) NAMES='[L40,L40S,RTX_6000Ada,A6000,RTX_A6000,A40]'; MINRAM=44000 ;;
  *)  NAMES='[RTX_3090,RTX_4090]'; MINRAM=23000 ;;
esac
OFFER=$(vastai search offers "gpu_name in $NAMES num_gpus=1 rentable=true verified=true disk_space>=150 inet_down>800 reliability>0.98 cuda_max_good>=12.8" -o 'dph' --raw \
  | python3 -c "import json,sys;d=[o for o in json.load(sys.stdin) if o['gpu_ram']>=$MINRAM];o=d[0];print(o['id']);print(o['gpu_name'],round(o['dph_total'],3),o['gpu_ram'],'dl',int(o['inet_down']),'host',o['host_id'],file=sys.stderr)")
vastai create instance "$OFFER" --image "$IMAGE" --disk 150 --ssh --direct --env "-e RLPAINT_MODEL=$MODEL" \
  --onstart-cmd 'cd /root/env && python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ.get(\"RLPAINT_MODEL\",\"Qwen/Qwen3.5-4B\"))" && touch /root/READY' --raw | grep -E 'new_contract|success'
echo "then: VAST_ID=<id> env/vast_run.sh ssh 'test -f /root/READY'; env/vast_run.sh train_trl ...; env/vast_run.sh destroy"
