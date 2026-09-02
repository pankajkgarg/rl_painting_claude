"""Build docs/gallery.html from out/rewards.jsonl + out/samples/*.png (base64-inlined)."""
import json, base64, re, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
recs = [json.loads(l) for l in (ROOT/os.environ.get("RLPAINT_OUT","out")/"rewards.jsonl").read_text().splitlines() if l.strip()]
samples = sorted((ROOT/os.environ.get("RLPAINT_OUT","out")/"samples").glob("*.png"))
def b64(p): return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
W, H = 640, 220
xs = [r["step"] for r in recs]; ys = [r["mean"] for r in recs]; ok = [r["ok"]/r["n"] for r in recs]
def poly(vals, ymax=1.0):
    n = max(len(vals),1)
    return " ".join(f"{20 + i*(W-40)/max(n-1,1):.1f},{H-20 - v/ymax*(H-40):.1f}" for i, v in enumerate(vals))
svg = f'''<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;background:#fafafa;border:1px solid #ddd">
<polyline fill="none" stroke="#c33" stroke-width="2" points="{poly(ys)}"/>
<polyline fill="none" stroke="#39c" stroke-width="1.5" stroke-dasharray="4 3" points="{poly(ok)}"/>
<text x="24" y="16" font-size="12" fill="#c33">mean reward</text><text x="120" y="16" font-size="12" fill="#39c">fraction rendered ok</text>
<text x="{W-60}" y="{H-4}" font-size="11">step {xs[-1] if xs else 0}</text></svg>'''
cards = []
for p in samples:
    m = re.match(r"s(\d+)_(best|rand)_([\d.]+)\.png", p.name)
    if not m: continue
    js = p.with_suffix(".js")
    code = js.read_text() if js.exists() else ""
    cards.append(f'<figure><img src="{b64(p)}" loading="lazy"><figcaption>step {int(m[1])} · {m[2]} · r={m[3]}</figcaption><details><summary>code</summary><pre>{code.replace("<","&lt;")}</pre></details></figure>')
html = f'''<!doctype html><meta charset=utf-8><title>RL paints p5.js — v0</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1100px}}figure{{display:inline-block;width:256px;margin:6px;vertical-align:top}}img{{width:256px;height:256px;border:1px solid #ccc}}figcaption{{font-size:12px;color:#444}}pre{{font-size:10px;white-space:pre-wrap;max-height:200px;overflow:auto;background:#f4f4f4}}</style>
<h1>GRPO teaches Qwen3-4B to paint with p5.js — v0</h1>
<p>Steps logged: {len(recs)} · last mean reward {ys[-1] if ys else 0:.3f} · first {ys[0] if ys else 0:.3f}</p>
{svg}
<h2>Samples (best + random rollout, every 5 steps)</h2>
{"".join(cards)}'''
(ROOT/"docs/gallery.html").write_text(html)
print("wrote docs/gallery.html", len(recs), "steps", len(cards), "images")
