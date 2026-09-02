"""GRPO: teach a small coding LLM to paint with p5.js. Unsloth + TRL.

Usage: python train.py [--model unsloth/Qwen3-4B] [--steps 100] [--gens 8] [--out /content/out]
Env: RLPAINT_VLLM=1 to try vLLM fast_inference (falls back to HF generate if it fails).
"""
import argparse, json, os, sys, time, re, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--gens", type=int, default=8)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--out", default="out")
ap.add_argument("--save_every", type=int, default=20)
ap.add_argument("--load_4bit", action="store_true")
args = ap.parse_args()

OUT = Path(args.out); (OUT / "samples").mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "rewards.jsonl", "a")

from train_prompt import SUBJECTS, SYSTEM, make_prompt

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from reward_v1 import score

use_vllm = os.environ.get("RLPAINT_VLLM", "0") == "1"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model, max_seq_length=args.max_new + 768, load_in_4bit=args.load_4bit, load_in_16bit=not args.load_4bit,
    fast_inference=use_vllm, max_lora_rank=args.lora_r, gpu_memory_utilization=0.5 if use_vllm else None,
)
model = FastLanguageModel.get_peft_model(
    model, r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=0,
)

# one row per step; GRPO samples `gens` completions per prompt
def render_prompt(subj):
    """Pre-render the chat template with thinking disabled and hand TRL a plain string."""
    msgs = make_prompt(subj)
    try:
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if "</think>" not in txt[-40:]:
        txt += "<think>\n\n</think>\n\n"  # Qwen3-style empty reasoning block = no-think
    return txt
_p0 = render_prompt(SUBJECTS[0]); print("PROMPT TAIL:", repr(_p0[-120:]), flush=True)
rows = [{"prompt": render_prompt(SUBJECTS[i % len(SUBJECTS)]), "subject": SUBJECTS[i % len(SUBJECTS)]}
        for i in range(args.steps * 4)]
ds = Dataset.from_list(rows)

STEP = {"n": 0}
def reward_fn(completions, subject, **kw):
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    t0 = time.time()
    rs, info = score(texts, list(subject))
    STEP["n"] += 1
    n = STEP["n"]
    ok = sum(d["ok"] for d in info)
    rec = dict(step=n, mean=float(sum(rs) / len(rs)), max=float(max(rs)), ok=ok, n=len(rs),
               errors=[d["error"] for d in info if not d["ok"]][:4],
               fail_snips=[texts[i][:160] for i, d in enumerate(info) if not d["ok"]][:2],
               mean_hps=float(sum(d["hps"] for d in info if d["ok"]) / max(ok, 1)) * 100,
               mean_sig=float(sum(d["sig"] for d in info if d["ok"]) / max(ok, 1)),
               subject=subject[0][:40],
               mean_len=float(sum(d["len"] for d in info) / len(info)), t=time.time() - t0)
    LOG.write(json.dumps(rec) + "\n"); LOG.flush()
    print(f"[reward] step={n} mean={rec['mean']:.3f} max={rec['max']:.3f} ok={ok}/{len(rs)} hps={rec['mean_hps']:.2f} sig={rec['mean_sig']:.3f} len={rec['mean_len']:.0f} {rec['t']:.0f}s", flush=True)
    # dump best + one random rendered sample every few steps
    if n % 5 == 1 or n <= 3:
        order = sorted(range(len(rs)), key=lambda i: -rs[i])
        for tag, i in (("best", order[0]), ("rand", random.choice(order))):
            if info[i]["png"]:
                (OUT / "samples" / f"s{n:04d}_{tag}_{rs[i]:.2f}.png").write_bytes(info[i]["png"])
                (OUT / "samples" / f"s{n:04d}_{tag}_{rs[i]:.2f}.js").write_text(texts[i])
    return rs

cfg = GRPOConfig(
    output_dir=str(OUT / "ckpt"), learning_rate=args.lr, adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1,
    warmup_ratio=0.05, lr_scheduler_type="cosine", optim="adamw_8bit", logging_steps=1,
    per_device_train_batch_size=args.gens, gradient_accumulation_steps=1, num_generations=args.gens,
    max_prompt_length=512, max_completion_length=args.max_new, max_steps=args.steps,
    save_steps=args.save_every, save_total_limit=2, report_to="none", bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(), temperature=1.0, use_vllm=use_vllm, beta=0.0,
    mask_truncated_completions=True,
)
trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[reward_fn], args=cfg, train_dataset=ds)
trainer.train()
model.save_pretrained(str(OUT / "lora_final"))
tokenizer.save_pretrained(str(OUT / "lora_final"))
print("TRAIN_DONE")
