"""Reward for p5.js painting rollouts. Zero API cost.

score(completions, subject) -> (rewards: list[float], info: list[dict])
Components (weights in W):
  gate      : static check + renders + non-blank            (hard gate: total=0 if fail)
  length    : soft band on code length
  clip_sim  : CLIP ViT-B/32 text<->image cosine to caption(s)
  aesthetic : LAION aesthetic-v2 linear head on CLIP ViT-B/32 (1..10 -> normalized)
"""
import io, os, sys, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import render_batch, extract_code

HERE = Path(__file__).resolve().parent
DEVICE = os.environ.get("RLPAINT_CLIP_DEVICE", "cpu")
W = dict(gate=0.10, length=0.05, clip_sim=0.30, aesthetic=0.55)
LEN_LO, LEN_HI = 400, 3500  # chars of code

_clip = None
def _load():
    global _clip
    if _clip is None:
        import open_clip
        model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=DEVICE)
        model.eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        head = torch.nn.Linear(512, 1)
        sd = torch.load(HERE / "aesthetic_vit_b_32.pth", map_location="cpu")
        head.load_state_dict(sd)
        head.eval().to(DEVICE)
        _clip = (model, pre, tok, head)
    return _clip

@torch.no_grad()
def image_scores(pngs, captions):
    """pngs: list[bytes]; captions: list[str] (pos caption first). returns (sim, aes) arrays."""
    model, pre, tok, head = _load()
    if not pngs:
        return np.zeros(0), np.zeros(0)
    ims = torch.stack([pre(Image.open(io.BytesIO(p)).convert("RGB")) for p in pngs]).to(DEVICE)
    f = model.encode_image(ims)
    f = f / f.norm(dim=-1, keepdim=True)
    t = model.encode_text(tok(captions).to(DEVICE))
    t = t / t.norm(dim=-1, keepdim=True)
    sims = (f @ t.T).float().cpu().numpy()  # [N, C]
    aes = head(f.float()).squeeze(-1).cpu().numpy()  # ~1..10
    return sims, aes

def length_reward(n):
    if n < LEN_LO: return max(0.0, n / LEN_LO)
    if n > LEN_HI: return max(0.0, 1 - (n - LEN_HI) / LEN_HI)
    return 1.0

def score(completions, subject="a watercolour painting of a pink hibiscus flower"):
    codes = [extract_code(c) for c in completions]
    rend = render_batch(codes)
    ok_idx = [i for i, r in enumerate(rend) if r["ok"]]
    captions = [subject, "a blank white image", "clip art, flat vector icon", "random noise, scribbles"]
    sims, aes = image_scores([rend[i]["png"] for i in ok_idx], captions)
    rewards, info = [], []
    for i, (code, r) in enumerate(zip(codes, rend)):
        d = dict(ok=r["ok"], error=r["error"], len=len(code), std=r["std"], png=r["png"])
        if not r["ok"]:
            d.update(clip_sim=0.0, aesthetic=0.0, length=0.0, reward=0.0)
        else:
            k = ok_idx.index(i)
            s = sims[k]
            # contrastive: positive minus best negative, squashed to ~[0,1]
            margin = float(s[0] - s[1:].max())
            clip_sim = float(np.clip(0.5 + margin * 5.0, 0, 1))
            a = float(np.clip((aes[k] - 3.0) / 4.0, 0, 1))  # 3..7 -> 0..1
            L = length_reward(len(code))
            rew = W["gate"] + W["length"] * L + W["clip_sim"] * clip_sim + W["aesthetic"] * a
            d.update(clip_sim=clip_sim, aesthetic=a, aes_raw=float(aes[k]), sim_raw=float(s[0]), length=L, reward=rew)
        rewards.append(d["reward"]); info.append(d)
    return rewards, info

if __name__ == "__main__":
    comps = [Path(f).read_text() for f in sys.argv[1:]]
    rs, inf = score(comps)
    for f, r, d in zip(sys.argv[1:], rs, inf):
        print(f"{f:28s} reward={r:.3f} ok={d['ok']} err={d['error']} sim={d.get('sim_raw',0):.3f} aes={d.get('aes_raw',0):.2f} len={d['len']}")
