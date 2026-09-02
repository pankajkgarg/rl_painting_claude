# RL-paints-p5js training image for vast.ai. Pins the stack that worked on 2026-09-02 (see log.md).
# Build: GitHub Actions (.github/workflows/image.yml) -> ghcr.io/<owner>/rlpaint:latest
# slim base: vllm pins its own torch (2.13+cu130), so a pytorch base image would just be 8GB of dead weight
FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
RUN apt-get update -qq && apt-get install -y -qq curl rsync git openssh-server libgomp1 build-essential && rm -rf /var/lib/apt/lists/*
# vllm 0.27.1 pins torch 2.13+cu130. TRL 1.12 targets transformers 5.x. (Unsloth 2026.8 pins trl<=0.24, which is
# incompatible with transformers 5.5+ and vllm 0.27, so this image uses plain TRL+PEFT: env/train_trl.py.)
RUN pip install "vllm==0.27.1" && \
    pip install "trl==1.12.0" peft playwright open_clip_torch sentencepiece hf_transfer flash-linear-attention && \
    pip install -U transformers
# flashinfer ships a py3.12-only annotation (array.array[int]); quote it so py3.11 can import it.
RUN F=$(python -c "import flashinfer,os;print(os.path.join(os.path.dirname(flashinfer.__file__),'comm','fd_exchange.py'))") && \
    sed -i -E 's/array\.array\[[A-Za-z_]+\]/"&"/g' "$F" && python -c "import ast; ast.parse(open('$F').read())"
RUN python -c "from trl.trainer.grpo_trainer import GRPOTrainer; import vllm" 
RUN python -m playwright install --with-deps chromium
# scorers baked in (2GB HPSv2.1 + 1.5GB SigLIP2); LLM weights are fetched at start (fast on vast, keeps image small)
RUN mkdir -p /root/env /root/out && \
    curl -sL -o /root/env/HPS_v2.1_compressed.pt https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt && \
    python -c "from transformers import AutoModel, AutoProcessor; AutoModel.from_pretrained('google/siglip2-base-patch16-224'); AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')"
COPY env/render.py env/reward_v1.py env/train.py env/train_prompt.py env/eval_samples.py env/animate.py env/p5.min.js /root/env/
WORKDIR /root/env
CMD ["sleep", "infinity"]
