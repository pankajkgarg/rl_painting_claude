#!/usr/bin/env bash
# Helpers for the vast.ai box. Usage: env/vast_run.sh {push|smoke|train|poll|pull|destroy} [args]
set -e
cd "$(dirname "$0")/.."
ID=${VAST_ID:-49613475}
read HOST PORT <<< $(vastai show instance $ID --raw | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['ssh_host'],d['ssh_port'])")
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT root@$HOST"
SCP="scp -o StrictHostKeyChecking=no -P $PORT"
case "$1" in
  push)   $SCP env/render.py env/reward_v1.py env/train.py env/p5.min.js temp/sk/good.js temp/sk/flatflower.js out_v0/samples/s0011_best_0.53.js root@$HOST:/root/env/ ;;
  smoke)  $SSH 'cd /root/env && RLPAINT_CONC=8 python reward_v1.py good.js flatflower.js s0011_best_0.53.js 2>&1 | grep -E "reward=|Error|Traceback"' ;;
  train)  shift; $SSH "cd /root/env && RLPAINT_CONC=8 RLPAINT_VLLM=${RLPAINT_VLLM:-0} nohup python train.py $* > /root/train.log 2>&1 &"; echo launched ;;
  poll)   $SSH 'grep -aE "reward\]|TRAIN_DONE|Traceback|Error|thinking disabled" /root/train.log | grep -v Warning | tail -${2:-8}; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader' ;;
  pull)   mkdir -p out/samples; $SCP root@$HOST:/root/out/rewards.jsonl out/ 2>/dev/null || true; rsync -a -e "ssh -o StrictHostKeyChecking=no -p $PORT" root@$HOST:/root/out/samples/ out/samples/ ; echo "pulled $(ls out/samples | wc -l | tr -d ' ') files" ;;
  ssh)    shift; $SSH "$@" ;;
  destroy) vastai destroy instance $ID ;;
  *) echo "usage: $0 {push|smoke|train|poll|pull|ssh|destroy}"; exit 1 ;;
esac
