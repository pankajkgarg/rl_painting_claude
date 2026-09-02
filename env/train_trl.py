"""GRPO with plain TRL + PEFT + vLLM colocate (no Unsloth). Needed for Qwen3.5 (multimodal arch that
Unsloth's fast_inference refuses). Same prompt/reward/logging as train.py.
Usage: python train_trl.py --model Qwen/Qwen3.5-4B --steps 150 --gens 8 --max_new 1400 --out /root/out
"""
import argparse, json, os, sys, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--gens", type=int, default=8); ap.add_argument("--max_new", type=int, default=1400)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--out", default="out"); ap.add_argument("--save_every", type=int, default=25)
ap.add_argument("--vllm_mem", type=float, default=0.25); ap.add_argument("--micro", type=int, default=2); ap.add_argument("--sleep", action="store_true")
args = ap.parse_args()
OUT = Path(args.out); (OUT / "samples").mkdir(parents=True, exist_ok=True); LOG = open(OUT / "rewards.jsonl", "a")
from train_prompt import SUBJECTS, SYSTEM, make_prompt
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from reward_v1 import score

tokenizer = AutoTokenizer.from_pretrained(args.model)
def render_prompt(subj):
    msgs = make_prompt(subj)
    try: txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if "</think>" not in txt[-40:]: txt += "<think>\n\n</think>\n\n"
    return txt
print("PROMPT TAIL:", repr(render_prompt(SUBJECTS[0])[-120:]), flush=True)
ds = Dataset.from_list([{"prompt": render_prompt(SUBJECTS[i % len(SUBJECTS)]), "subject": SUBJECTS[i % len(SUBJECTS)]} for i in range(args.steps * 4)])

STEP = {"n": 0}
def reward_fn(completions, subject, **kw):
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    t0 = time.time(); rs, info = score(texts, list(subject)); STEP["n"] += 1; n = STEP["n"]; ok = sum(d["ok"] for d in info)
    rec = dict(step=n, mean=float(sum(rs)/len(rs)), max=float(max(rs)), ok=ok, n=len(rs),
               errors=[d["error"] for d in info if not d["ok"]][:4], fail_snips=[texts[i][:160] for i, d in enumerate(info) if not d["ok"]][:2],
               mean_hps=float(sum(d["hps"] for d in info if d["ok"]) / max(ok, 1)) * 100,
               mean_sig=float(sum(d["sig"] for d in info if d["ok"]) / max(ok, 1)), subject=subject[0][:40],
               mean_len=float(sum(d["len"] for d in info) / len(info)), t=time.time() - t0)
    LOG.write(json.dumps(rec) + "\n"); LOG.flush()
    print(f"[reward] step={n} mean={rec['mean']:.3f} max={rec['max']:.3f} ok={ok}/{len(rs)} hps={rec['mean_hps']:.2f} sig={rec['mean_sig']:.3f} len={rec['mean_len']:.0f} {rec['t']:.0f}s", flush=True)
    if n % 5 == 1 or n <= 3:
        order = sorted(range(len(rs)), key=lambda i: -rs[i])
        for tag, i in (("best", order[0]), ("rand", random.choice(order))):
            if info[i]["png"]:
                (OUT / "samples" / f"s{n:04d}_{tag}_{rs[i]:.2f}.png").write_bytes(info[i]["png"]); (OUT / "samples" / f"s{n:04d}_{tag}_{rs[i]:.2f}.js").write_text(texts[i])
    return rs

peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
cfg_kwargs = dict(
    output_dir=str(OUT / "ckpt"), learning_rate=args.lr, adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1,
    warmup_steps=5, lr_scheduler_type="cosine", logging_steps=1, per_device_train_batch_size=args.micro,
    gradient_accumulation_steps=max(1, args.gens // args.micro), num_generations=args.gens, max_prompt_length=512, max_completion_length=args.max_new,
    max_steps=args.steps, save_steps=args.save_every, save_total_limit=2, report_to="none", bf16=True,
    gradient_checkpointing=True, temperature=1.0, beta=0.0, mask_truncated_completions=True,
    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_mem, vllm_max_model_length=args.max_new + 768,
    vllm_enable_sleep_mode=args.sleep,
)
_fields = GRPOConfig.__dataclass_fields__
dropped = [k for k in cfg_kwargs if k not in _fields]
print("GRPOConfig: dropping unsupported keys:", dropped, flush=True)
cfg = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in _fields})
trainer = GRPOTrainer(model=args.model, processing_class=tokenizer, reward_funcs=[reward_fn], args=cfg, train_dataset=ds, peft_config=peft_cfg)
trainer.train()
trainer.model.save_pretrained(str(OUT / "lora_final")); tokenizer.save_pretrained(str(OUT / "lora_final"))
print("TRAIN_DONE")
