# Notes

## Anchor
- current default: v1.1 = Qwen3-4B-Instruct-2507 LoRA GRPO + vLLM on vast A6000, reward_v1 (gates + rank HPSv2.1 .75 + rank SigLIP2 .25), 5 subjects · sealed: v0 reward dead (CLIP-B/32); Qwen3.5-9B unusable (no vLLM, 184 s/step); Colab free unusable (reclaimed twice)
- alias map: "the painting RL" = this project; "surya's method" = GRPO + render + judge (surya.website/rling-qwen-to-paint-with-code)
- active threads: [v1-a6000] Qwen3.5-9B + reward_v1 on vast instance 49613475 (RTX A6000, $0.455/hr) · Colab free abandoned (reclaimed twice)
- next: (1) local red-team test of reward_v1 (2) v1 run on rlpaint2, 60 steps (3) progression.mp4 + gallery (4) if user funds vast.ai: 9B + VLM pairwise judge
- standing gates: budget cap $10 total; reward must gate text()/images so CLIP can't be hacked with words

## Overview
Replicate "RL a coding model to paint with javascript" (kickingkeys, Aug 2026): GRPO on a coding LLM that emits
p5.js sketches; rollouts rendered headlessly; reward = compile gate + length + aesthetic model + judge.
Goal: cheapest possible proof of concept.

## Current status
2026-09-02 14:00: v1 (Qwen3-8B + vLLM, reward_v1, 100 steps, 53 min, ~$0.45) ran on vast A6000 but images degenerate:
blank-canvas failures from HSB alpha trap, pixel-array garbage, naive concentric ellipses; HPS +1 pt only. v1.1 launched:
Qwen3-4B-Instruct-2507 + vLLM, gate forbids colorMode/pixels, technique prompt, lr 2e-5, 150 steps. Total spend ~$1.2 of $7.29.
Instance 49613475 still up ($0.48/hr) - destroy when done.

## Architecture
- Decision: plain p5.js 2D, not p5.brush. Why: p5.brush needs WEBGL; headless software GL is a rabbit hole. See log 2026-09-01.
- Decision: no API judge in v0. Why: $10 cap; CLIP+aesthetic is free. Forbid text() so CLIP can't be word-hacked.
- Decision: Colab free T4 via `colab --auth=adc` CLI (works; oauth2 default prompts). vast.ai 3090 ~$0.11/hr is the paid fallback (balance $0).

## Invariants & gotchas
- `colab` CLI: always pass `--auth=adc`; always `colab --auth=adc stop -s rlpaint` when done.
- Long jobs on VM: nohup subprocess + poll log, never hold `colab exec` open.
