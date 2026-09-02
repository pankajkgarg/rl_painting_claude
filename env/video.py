"""Make docs/progression.mp4 (best rollout per logged step, with step/reward caption) and
docs/grid.mp4 (top-K paintings tiled) from out/samples/*.png using ffmpeg."""
import re, subprocess, sys, shutil, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parent.parent
S = ROOT / os.environ.get("RLPAINT_OUT","out") / "samples"; T = ROOT / "temp/frames"; D = ROOT / "docs"
shutil.rmtree(T, ignore_errors=True); T.mkdir(parents=True); D.mkdir(exist_ok=True)
try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
except Exception: font = ImageFont.load_default()
def parse(p):
    m = re.match(r"s(\d+)_(best|rand)_([\d.]+)\.png", p.name); return (int(m[1]), m[2], float(m[3])) if m else None
import json
recs = {r["step"]: r for r in (json.loads(l) for l in (ROOT / os.environ.get("RLPAINT_OUT","out") / "rewards.jsonl").read_text().splitlines() if l.strip())}
LANE = os.environ.get("RLPAINT_SUBJECT", "")  # substring filter, e.g. "hibiscus"; empty = all subjects
best = sorted([(parse(p), p) for p in S.glob("s*_best_*.png") if parse(p)])
best = [(k, p) for k, p in best if not LANE or LANE in recs.get(k[0], {}).get("subject", "")]
for i, ((step, _, r), p) in enumerate(best):
    rec = recs.get(step, {}); hps = rec.get("mean_hps"); subj = rec.get("subject", "")[:34]
    im = Image.open(p).convert("RGB").resize((512, 512))
    canvas = Image.new("RGB", (512, 560), "white"); canvas.paste(im, (0, 0))
    cap = f"step {step:3d}" + (f"   HPS {hps:.1f}" if hps else "") + (f"   {subj}" if not LANE else "")
    ImageDraw.Draw(canvas).text((12, 520), cap, fill="black", font=font)
    canvas.save(T / f"f{i:04d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1.25", "-i", str(T / "f%04d.png"),
                "-vf", "format=yuv420p,scale=512:560", "-c:v", "libx264", "-crf", "20", str(D / ("progression_" + LANE + ".mp4" if LANE else "progression.mp4"))], check=True)
# grid: top 9 by reward across all samples (best+rand), 3x3, slow zoom
allp = sorted([(parse(p), p) for p in S.glob("s*_*.png") if parse(p)], key=lambda x: -x[0][2])[:9]
G = Image.new("RGB", (3 * 512, 3 * 512), "white")
for k, (_, p) in enumerate(allp):
    G.paste(Image.open(p).convert("RGB").resize((512, 512)), ((k % 3) * 512, (k // 3) * 512))
G.save(T / "grid.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(T / "grid.png"), "-t", "8",
                "-vf", "scale=1024:1024,zoompan=z='min(zoom+0.0008,1.15)':d=200:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1024x1024:fps=25,format=yuv420p",
                "-c:v", "libx264", "-crf", "20", str(D / "grid.mp4")], check=True)
print("wrote", D / "progression.mp4", f"({len(best)} frames)", "and", D / "grid.mp4")
