#!/usr/bin/env bash
# Rent a GPU on vast.ai with the pinned onstart. Usage: GPU=80 env/vast_launch.sh   (GPU=48 default)
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-48}
if [ "$GPU" = "80" ]; then NAMES='[A100_SXM4,A100_PCIE,H100_SXM,H100_PCIE,H100_NVL,H200]'; MINRAM=79000; else NAMES='[L40,L40S,RTX_6000Ada,A6000,RTX_A6000,A40]'; MINRAM=44000; fi
OFFER=$(vastai search offers "gpu_name in $NAMES num_gpus=1 rentable=true verified=true disk_space>=80 inet_down>300 reliability>0.97 cuda_max_good>=12.8" -o 'dph' --raw \
  | python3 -c "import json,sys;d=[o for o in json.load(sys.stdin) if o['gpu_ram']>=$MINRAM];o=d[0];print(o['id']);print(o['gpu_name'],o['dph_total'],o['gpu_ram'],file=sys.stderr)")
vastai create instance "$OFFER" --image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel --disk 80 --ssh --direct --onstart env/vast_onstart.sh --raw
echo "then: VAST_ID=<id> env/vast_run.sh push; wait for /root/READY; env/vast_run.sh train ...; env/vast_run.sh destroy"
