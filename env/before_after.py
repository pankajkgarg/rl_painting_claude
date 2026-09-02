"""Before/after grid from eval outputs: per subject, top-K by HPS for base (left) vs trained (right).
Usage: python before_after.py out/eval2 docs/before_after.png [K=3]"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
E = Path(sys.argv[1]); OUT = sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26); small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except Exception: font = small = ImageFont.load_default()
def top(arm, subj):
    rs = [r for r in json.loads((E/arm/"scores.json").read_text()) if r["subject"] == subj and r["ok"]]
    return sorted(rs, key=lambda r: -r["hps"])[:K]
subjects = [r["subject"] for r in json.loads((E/"base"/"scores.json").read_text())]
subjects = list(dict.fromkeys(subjects))
T = 256; GAP = 8; HDR = 70; LBL = 30
W = 2*K*T + (2*K+1)*GAP + 40; H = HDR + len(subjects)*(T+LBL+GAP)
G = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(G)
d.text((GAP, 12), "base model", fill="#666", font=font); d.text((K*T + (K+1)*GAP + 40, 12), "after 150 GRPO steps", fill="#c33", font=font)
d.text((GAP, 44), "Qwen3-4B-Instruct-2507, best 3 of 12 samples per prompt, scored by HPSv2.1", fill="#999", font=small)
y = HDR
for s in subjects:
    d.text((GAP, y), s, fill="#333", font=small); y += LBL
    for col, arm in enumerate(["base", "trained"]):
        x0 = GAP + col*(K*T + K*GAP + 40)
        for i, r in enumerate(top(arm, s)):
            im = Image.open(E/arm/(r["file"]+".png")).convert("RGB").resize((T, T)); G.paste(im, (x0 + i*(T+GAP), y))
            d.text((x0 + i*(T+GAP) + 4, y + T - 22), f"HPS {r['hps']*100:.1f}", fill="white", font=small)
    y += T + GAP
G.save(OUT); print("wrote", OUT, G.size)
# also: overall stats
for arm in ["base","trained"]:
    rs = json.loads((E/arm/"scores.json").read_text()); ok=[r for r in rs if r["ok"]]
    print(f"{arm:8s} ok={len(ok)}/{len(rs)} mean_hps={sum(r['hps'] for r in ok)/len(ok)*100:.2f} best_hps={max(r['hps'] for r in ok)*100:.2f}")
