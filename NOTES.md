# Notes

## Anchor
- current default: v1.2 = Qwen3.5-9B (v1.1 = Qwen3-4B-Instruct-2507 proven) LoRA GRPO + vLLM on vast A6000, reward_v1 (gates + rank HPSv2.1 .75 + rank SigLIP2 .25), 5 subjects · sealed: v0 reward dead (CLIP-B/32); Qwen3.5-9B unusable (no vLLM, 184 s/step); Colab free unusable (reclaimed twice)
- alias map: "the painting RL" = this project; "surya's method" = GRPO + render + judge (surya.website/rling-qwen-to-paint-with-code)
- active threads: [v1-a6000] Qwen3.5-9B + reward_v1 on vast instance 49613475 (RTX A6000, $0.455/hr) · Colab free abandoned (reclaimed twice)
- next: (1) local red-team test of reward_v1 (2) v1 run on rlpaint2, 60 steps (3) progression.mp4 + gallery (4) if user funds vast.ai: 9B + VLM pairwise judge
- standing gates: budget cap $10 total; reward must gate text()/images so CLIP can't be hacked with words; USE CURRENT-GENERATION MODELS (user: Jul-2025 models are archaic; check release dates before picking)

## Overview
Replicate "RL a coding model to paint with javascript" (kickingkeys, Aug 2026): GRPO on a coding LLM that emits
p5.js sketches; rollouts rendered headlessly; reward = compile gate + length + aesthetic model + judge.
Goal: cheapest possible proof of concept.

## Current status
2026-09-02 17:00: v1.1 (Qwen3-4B-Instruct-2507 + vLLM, reward_v1, 150 steps, 50 min, ~$0.40) is the first run that works:
HPS 10.7->13.6, render ok 79%->96%; base-vs-trained eval (12/subject): ~10.0 vs ~13.4 HPS. User objects to the
Jul-2025 model; v1.2 = Qwen3.5-9B + vLLM 0.27.1 (supported after all) queued on the same A6000. Spend $1.63 of $7.29.
LoRAs backed up to out/lora_final, out_v1/lora_final. Repo: github.com/pankajkgarg/rl_painting_claude (private);
workflow file held in temp/hold until `gh auth refresh -s workflow`. Deliverables pending: tweet video, writeup.

## Architecture
- Decision: plain p5.js 2D, not p5.brush. Why: p5.brush needs WEBGL; headless software GL is a rabbit hole. See log 2026-09-01.
- Decision: no API judge in v0. Why: $10 cap; CLIP+aesthetic is free. Forbid text() so CLIP can't be word-hacked.
- Decision: Colab free T4 via `colab --auth=adc` CLI (works; oauth2 default prompts). vast.ai 3090 ~$0.11/hr is the paid fallback (balance $0).

## Invariants & gotchas
- `colab` CLI: always pass `--auth=adc`; always `colab --auth=adc stop -s rlpaint` when done.
- Long jobs on VM: nohup subprocess + poll log, never hold `colab exec` open.
