# RL-paints-p5js training image for vast.ai. Pins the stack that worked on 2026-09-02 (see log.md).
# Build: GitHub Actions (.github/workflows/image.yml) -> ghcr.io/<owner>/rlpaint:latest
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update -qq && apt-get install -y -qq curl rsync openssh-server && rm -rf /var/lib/apt/lists/*
# vllm 0.27.1 (0.28 breaks unsloth_zoo's bridge); it brings torch 2.13+cu130. Then the rest on top.
RUN pip install "vllm==0.27.1" && \
    pip install unsloth==2026.8.22 unsloth_zoo==2026.8.17 trl==0.24.0 playwright open_clip_torch \
        sentencepiece hf_transfer flash-linear-attention && \
    pip install "transformers==5.5.0"
# flashinfer ships a py3.12-only annotation (array.array[int]); quote it so py3.11 can import it.
RUN F=$(python -c "import flashinfer,os;print(os.path.join(os.path.dirname(flashinfer.__file__),'comm','fd_exchange.py'))") && \
    sed -i -E 's/array\.array\[[A-Za-z_]+\]/"&"/g' "$F" && python -c "import ast,sys; ast.parse(open('$F').read())"
# TRL 0.24 vs transformers 5.5: _is_package_available now returns a tuple (truthy) -> TRL thinks every optional dep exists.
RUN F=$(python -c "import trl,os;print(os.path.join(os.path.dirname(trl.__file__),'import_utils.py'))") && \
    python - "$F" <<'PY'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("from transformers.utils.import_utils import _is_package_available\n",
"from transformers.utils.import_utils import _is_package_available as _tf_is_package_available\n\ndef _is_package_available(pkg_name, return_version=False):\n    r = _tf_is_package_available(pkg_name, return_version=True)\n    return r if return_version else bool(r[0])\n",1)
open(p,"w").write(s)
PY
RUN python -c "from trl.trainer.grpo_trainer import GRPOTrainer" 2>&1 | tail -1
RUN python -m playwright install --with-deps chromium
# scorers baked in (2GB HPSv2.1 + 1.5GB SigLIP2); LLM weights are fetched at start (fast on vast, keeps image small)
RUN mkdir -p /root/env /root/out && \
    curl -sL -o /root/env/HPS_v2.1_compressed.pt https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt && \
    python -c "from transformers import AutoModel, AutoProcessor; AutoModel.from_pretrained('google/siglip2-base-patch16-224'); AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')"
COPY env/render.py env/reward_v1.py env/train.py env/train_prompt.py env/eval_samples.py env/animate.py env/p5.min.js /root/env/
WORKDIR /root/env
CMD ["sleep", "infinity"]
