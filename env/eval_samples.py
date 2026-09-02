"""Before/after evidence: sample N sketches per subject from the base model and from a LoRA checkpoint
(vLLM), render + score with reward_v1, save every PNG/JS + scores.json. Run on the GPU box.
Usage: python eval_samples.py --lora /root/out/lora_final --n 12 --out /root/eval
"""
import argparse, json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507")
ap.add_argument("--lora", default=None); ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", default="/root/eval")
ap.add_argument("--max_new", type=int, default=1400); ap.add_argument("--tag", default=None)
a = ap.parse_args()

def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    import train_prompt as tp
    from reward_v1 import score, hps_scores
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, enable_lora=bool(a.lora), max_lora_rank=64, gpu_memory_utilization=0.6, max_model_len=2400, dtype="bfloat16")
    sp = SamplingParams(temperature=1.0, max_tokens=a.max_new, n=a.n)
    tag = a.tag or ("trained" if a.lora else "base")
    out = Path(a.out) / tag; out.mkdir(parents=True, exist_ok=True)
    lreq = LoRARequest("lora", 1, a.lora) if a.lora else None
    results = []
    for si, subj in enumerate(tp.SUBJECTS):
        prompt = tok.apply_chat_template(tp.make_prompt(subj), tokenize=False, add_generation_prompt=True)
        outs = llm.generate([prompt], sp, lora_request=lreq)[0].outputs
        texts = [o.text for o in outs]
        rs, info = score(texts, subj)
        for k, (t, d) in enumerate(zip(texts, info)):
            base = out / f"s{si}_{k:02d}"
            base.with_suffix(".js").write_text(t)
            if d["png"]: base.with_suffix(".png").write_bytes(d["png"])
            results.append(dict(subject=subj, idx=k, file=base.name, ok=d["ok"], error=d["error"], hps=d["hps"], sig=d["sig"], len=d["len"]))
        ok = [r for r in results if r["subject"] == subj and r["ok"]]
        print(f"[{tag}] {subj[:40]:40s} ok={len(ok)}/{a.n} mean_hps={sum(r['hps'] for r in ok)/max(len(ok),1)*100:.2f}", flush=True)
    (out / "scores.json").write_text(json.dumps(results, indent=1))
    print("EVAL_DONE", tag)

if __name__ == "__main__":
    main()
