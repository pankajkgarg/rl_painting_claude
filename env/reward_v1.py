"""v1 reward: hard gates + within-group rank of HPSv2.1 and SigLIP2 (per Codex research, docs/reward_model_research.md).

R = -1                       if static/render/blank gate fails
  = 0.75*rank(HPSv2.1) + 0.25*rank(SigLIP2)   otherwise, ranks in [0,1] within each prompt group.
Raw HPS / SigLIP means are logged so absolute progress is visible even though the training signal is relative.
"""
import io, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import render_batch, extract_code

HERE = Path(__file__).resolve().parent
DEVICE = os.environ.get("RLPAINT_SCORER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MB = int(os.environ.get("RLPAINT_SCORER_MB", "2"))
W_H, W_S = 0.75, 0.25
_m = {}

def _load():
    if _m: return _m
    import open_clip
    from transformers import AutoModel, AutoProcessor
    hps, _, pre = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None, device="cpu")
    ck = torch.load(HERE / "HPS_v2.1_compressed.pt", map_location="cpu")
    hps.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    hps = hps.to(DEVICE, dtype=DTYPE).eval()
    _m.update(hps=hps, hps_pre=pre, hps_tok=open_clip.get_tokenizer("ViT-H-14"))
    sid = "google/siglip2-base-patch16-224"
    _m["sig"] = AutoModel.from_pretrained(sid, dtype=DTYPE).eval().to(DEVICE)
    _m["sig_proc"] = AutoProcessor.from_pretrained(sid)
    return _m

@torch.no_grad()
def hps_scores(imgs, prompt):
    m = _load(); out = []
    t = m["hps"].encode_text(m["hps_tok"]([prompt]).to(DEVICE)); t = t / t.norm(dim=-1, keepdim=True)
    for i in range(0, len(imgs), MB):
        x = torch.stack([m["hps_pre"](im) for im in imgs[i:i+MB]]).to(DEVICE, dtype=DTYPE)
        f = m["hps"].encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        out += (f @ t.T).squeeze(-1).float().cpu().tolist()
    return out  # cosine; hpsv2 package reports 100x this

@torch.no_grad()
def sig_scores(imgs, prompt):
    m = _load(); out = []
    for i in range(0, len(imgs), MB):
        inp = m["sig_proc"](text=[prompt], images=imgs[i:i+MB], padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
        if DEVICE == "cuda": inp["pixel_values"] = inp["pixel_values"].to(DTYPE)
        out += m["sig"](**inp).logits_per_image[:, 0].float().cpu().tolist()  # raw logits: sigmoid saturates near 0
    return out

def rank01(v):
    v = np.asarray(v, dtype=np.float64); n = len(v)
    if n <= 1: return np.ones(n)
    order = v.argsort(); r = np.empty(n); r[order] = np.arange(n)
    # average ties
    for u in np.unique(v):
        idx = np.where(v == u)[0]
        if len(idx) > 1: r[idx] = r[idx].mean()
    return r / (n - 1)

def score(completions, subject):
    """completions: list[str]; subject: str or list[str] (one per completion). Groups by subject."""
    subs = [subject] * len(completions) if isinstance(subject, str) else list(subject)
    codes = [extract_code(c) for c in completions]
    rend = render_batch(codes)
    info = [dict(ok=r["ok"], error=r["error"], len=len(c), std=r["std"], png=r["png"], hps=None, sig=None, reward=-1.0)
            for c, r in zip(codes, rend)]
    for s in set(subs):
        idx = [i for i in range(len(subs)) if subs[i] == s and rend[i]["ok"]]
        if not idx: continue
        imgs = [Image.open(io.BytesIO(rend[i]["png"])).convert("RGB") for i in idx]
        h = hps_scores(imgs, s); g = sig_scores(imgs, s)
        rh, rg = rank01(h), rank01(g)
        for k, i in enumerate(idx):
            info[i].update(hps=float(h[k]), sig=float(g[k]), reward=float(W_H * rh[k] + W_S * rg[k]))
    return [d["reward"] for d in info], info

if __name__ == "__main__":
    comps = [Path(f).read_text() for f in sys.argv[1:]]
    rs, inf = score(comps, "a watercolour painting of a pink hibiscus flower")
    for f, r, d in zip(sys.argv[1:], rs, inf):
        print(f"{f.split('/')[-1]:26s} reward={r:+.3f} ok={d['ok']} err={d['error']} hps={d['hps'] if d['hps'] is None else round(d['hps']*100,2)} sig={d['sig'] if d['sig'] is None else round(d['sig'],4)} len={d['len']}")
