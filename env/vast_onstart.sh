#!/bin/bash
# vast.ai onstart for pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel. Pinned stack + patches that worked on 2026-09-02.
# Touches /root/READY when done. Log: /root/onstart.log
exec > /root/onstart.log 2>&1
set -x
apt-get update -qq && apt-get install -y -qq curl rsync >/dev/null
pip install -q "vllm==0.27.1"                                   # 0.28 breaks unsloth_zoo; brings torch 2.13+cu130
pip install -q unsloth==2026.8.22 unsloth_zoo==2026.8.17 trl==0.24.0 playwright open_clip_torch sentencepiece hf_transfer flash-linear-attention
pip install -q "transformers==5.5.0"
# flashinfer: py3.12-only annotation
F=$(python -c "import flashinfer,os;print(os.path.join(os.path.dirname(flashinfer.__file__),'comm','fd_exchange.py'))"); sed -i -E 's/array\.array\[[A-Za-z_]+\]/"&"/g' "$F"
# TRL 0.24 vs transformers 5.5: _is_package_available returns a tuple
F=$(python -c "import trl,os;print(os.path.join(os.path.dirname(trl.__file__),'import_utils.py'))")
python - "$F" <<'PY'
import sys; p=sys.argv[1]; s=open(p).read()
if "_tf_is_package_available" not in s:
    s=s.replace("from transformers.utils.import_utils import _is_package_available\n",
    "from transformers.utils.import_utils import _is_package_available as _tf_is_package_available\n\ndef _is_package_available(pkg_name, return_version=False):\n    r = _tf_is_package_available(pkg_name, return_version=True)\n    return r if return_version else bool(r[0])\n",1)
    open(p,"w").write(s)
PY
python -c "from trl.trainer.grpo_trainer import GRPOTrainer; print('TRL_IMPORT_OK')"
python -m playwright install --with-deps chromium
mkdir -p /root/env /root/out
curl -sL -o /root/env/HPS_v2.1_compressed.pt https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
python - <<'PY'
from transformers import AutoModel, AutoProcessor
AutoModel.from_pretrained("google/siglip2-base-patch16-224"); AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
from huggingface_hub import snapshot_download
import os
snapshot_download(os.environ.get("RLPAINT_MODEL", "Qwen/Qwen3.5-9B"))
PY
pip list 2>/dev/null | grep -iE '^(torch|transformers|trl|unsloth|vllm|playwright|open_clip_torch|flashinfer) '
nvidia-smi --query-gpu=name,memory.total --format=csv
touch /root/READY
