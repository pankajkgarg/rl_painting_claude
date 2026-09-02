# RL a coding LLM to paint with p5.js (proof of concept)

Replicates the "you can just RL a coding model to paint with javascript" idea (GRPO on a small coding LLM
whose completions are p5.js sketches; each rollout is rendered headlessly and scored).

## Pieces
- `env/render.py`  Playwright + vendored p5.js 1.11.3. Static forbid-list (text(), images, fetch, DOM...),
  fixed 512x512 canvas, timeout, blank detection. `render_batch(codes)`.
- `env/reward.py`  gate 0.10 + length 0.05 + CLIP ViT-B/32 contrastive similarity 0.30 + LAION aesthetic head 0.55.
  No API calls; runs on CPU.
- `env/train.py`   Unsloth + TRL GRPO, LoRA r16, `unsloth/Qwen3-4B-Instruct-2507`. Logs `out/rewards.jsonl`,
  dumps best+random rendered PNG every 5 steps to `out/samples/`.
- `env/sync.sh`    pull results from the Colab VM. `env/gallery.py` builds `docs/gallery.html`.
- `env/vast_onstart.sh` + `env/vast_run.sh` {push,smoke,train,poll,pull,destroy}: the vast.ai path (used for v1, Qwen3.5-9B on an RTX A6000).

## Run (free Colab T4)
```bash
colab --auth=adc new -s rlpaint --gpu T4
# upload env/*, install unsloth trl playwright open_clip_torch + playwright chromium deps, then on the VM:
cd /content/env && nohup python train.py --steps 80 --gens 8 --max_new 800 --out /content/out > /content/train.log 2>&1 &
```
Always `colab --auth=adc stop -s rlpaint` afterwards.

## Local test
```bash
uv venv .venv -p 3.12 && . .venv/bin/activate && uv pip install playwright pillow numpy torch torchvision open_clip_torch
python -m playwright install chromium
python env/reward.py temp/sk/good.js temp/sk/flatflower.js temp/sk/texthack.js
```
