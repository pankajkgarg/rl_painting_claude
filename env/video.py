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
best = sorted([(parse(p), p) for p in S.glob("s*_best_*.png") if parse(p)])
# progression: one frame per logged step, hold 0.8s each -> 1.25 fps then upsample w/ crossfade via minterpolate-free approach: just repeat
for i, ((step, _, r), p) in enumerate(best):
    im = Image.open(p).convert("RGB").resize((512, 512))
    canvas = Image.new("RGB", (512, 560), "white"); canvas.paste(im, (0, 0))
    ImageDraw.Draw(canvas).text((12, 520), f"GRPO step {step:3d}   reward {r:.2f}", fill="black", font=font)
    canvas.save(T / f"f{i:04d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1.25", "-i", str(T / "f%04d.png"),
                "-vf", "format=yuv420p,scale=512:560", "-c:v", "libx264", "-crf", "20", str(D / "progression.mp4")], check=True)
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
