# Teaching a 4B coding model to paint with p5.js, for about $2

*Replicating "you can just RL a coding model to paint with javascript" (Surya Narreddi, Aug 2026) as cheaply as possible.*

## The idea

A small coding LLM is prompted "paint a watercolour hibiscus" and answers with a complete p5.js sketch.
The sketch is rendered headlessly to a PNG, the PNG is scored, and GRPO nudges the model toward
whatever scored well. No image model anywhere in the loop: the artwork is code, so it stays editable.

Surya's recipe: GRPO, p5.brush sketches rendered with Puppeteer, reward = 0.05 compile + 0.05 length +
0.30 HPSv3 + 0.60 pairwise VLM judge against 581 hand-rated reference paintings. A second replicator
(gui dávid) used Qwen-3.5-35B for 200 steps and stopped "because money".

## What I built

| Piece | File | Notes |
|---|---|---|
| Renderer | `env/render.py` | Playwright + vendored p5.js 1.11.3, fixed 512x512, per-sketch timeout, blank detection, static forbid-list (text, images, fetch, DOM, colorMode, pixel arrays) |
| Reward v0 | `env/reward.py` | CLIP ViT-B/32 similarity + LAION aesthetic head. Dead end. |
| Reward v1 | `env/reward_v1.py` | Hard gate (-1) then within-group rank of HPSv2.1 (0.75) + SigLIP2 (0.25). Zero API cost. |
| Trainer | `env/train.py` | Unsloth + TRL GRPO, LoRA r16, 8 rollouts/prompt, 5 flower subjects |
| Evidence | `env/eval_samples.py`, `env/video.py`, `env/animate.py` | base-vs-trained sampling, progression video, "painting draws itself" clip |

## The runs

| Run | Model | Compute | Reward | Steps | Result |
|---|---|---|---|---|---|
| v0 | Qwen3-4B-Instruct-2507 | free Colab T4 (HF generate, 3 min/step) | CLIP-B/32 + aesthetic | 22 (VM reclaimed) | Flat. Every rollout = concentric circles on a disc. Reward could not tell a pink disc from a flower. |
| v1 | Qwen3-8B | vast.ai RTX A6000 + vLLM (32 s/step) | HPSv2.1 + SigLIP2 ranks | 100 | HPS +1 pt. Most rollouts blank: HSB colorMode with alpha scaled 0-100, then alpha 0.3. Naive concentric ellipses. |
| v1.1 | Qwen3-4B-Instruct-2507 | same box (20 s/step) | same + forbid colorMode/pixels, technique prompt, lr 2e-5 | 150 | HPS 10.7 -> 13.6, render success 79% -> 96%. Layered petals, textured backgrounds, per-subject variety. |

EVAL_PLACEHOLDER

## What actually mattered

1. **The reward is the whole game.** CLIP-B/32 + an aesthetic head gave within-group reward std of 0.04, so GRPO had no gradient. HPSv2.1 ranked a hand-written watercolour (17.6) above flat clip-art (16.9) above the v0 discs (12-13) above scribbles (1.3). Ranking within the 8-rollout group, as the research suggested, makes the scale irrelevant.
2. **Gate what the model can hack, don't reward proxies.** text() would let the model write "hibiscus" for CLIP. colorMode's alpha-scale trap produced blank canvases that a naive reward would have ignored.
3. **Base model taste beats parameter count here.** Qwen3-8B (non-thinking) wrote worse sketches than Qwen3-4B-Instruct-2507. Qwen3.5-9B was the best writer but has no vLLM support yet: 184 s/step, unaffordable.
4. **Thinking mode silently kills RL runs.** Qwen3.5 emitted a "Thinking Process:" preamble that ate the whole token budget. TRL 0.24 has no chat_template_kwargs, so the fix was pre-rendering the prompt string with enable_thinking=False.
5. **Generation speed is everything.** HF generate: 137-207 s/step. vLLM: 20-32 s/step. A 150-step run went from a 7-hour Colab gamble to 50 minutes for 40 cents.

## Cost

COST_PLACEHOLDER

## Reproduce

See README.md. Docker image (pinned vllm 0.27.1, unsloth 2026.8.22, HPSv2.1, SigLIP2, Chromium) builds from `docker/Dockerfile`; `env/vast_launch_image.sh` rents a 45GB+ GPU with it and is training-ready in a few minutes.

## Next

Pairwise judge against a hand-rated reference pool (the 0.60-weight term in Surya's recipe) using Qwen3-VL-2B on the same 48GB box; p5.brush instead of raw p5 once WEBGL rendering is sorted in headless Chromium; more subjects.
