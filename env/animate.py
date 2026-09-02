"""Render a p5.js sketch 'painting itself': re-run draw() with an increasing primitive budget
(deterministic seeds), capture each stage, and encode an mp4 with ffmpeg.
Usage: python animate.py sketch.js out.mp4 [frames=90] [seconds=6]
"""
import asyncio, re, sys, subprocess, shutil, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import P5, SIZE, _force_size, extract_code

PRIMS = ["ellipse","circle","rect","line","bezier","curve","arc","point","triangle","quad","square","endShape","vertex","curveVertex","bezierVertex","image","background"]
HTML = """<!doctype html><html><head><meta charset=utf-8><style>body{margin:0;background:#fff}</style>
<script>%s</script>
<script>
window.__budget=1e9; window.__count=0; window.__total=0; window.__err=null; window.__stage=0;
window.addEventListener('error',e=>{ if(!(e.error&&e.error.__stop)) window.__err=String(e.message||e.error)});
function __wrap(){ for(const n of %s){ const f=window[n]; if(typeof f!=='function') continue;
  window[n]=function(){ window.__count++; window.__total++; if(window.__count>window.__budget){ const s=new Error('budget'); s.__stop=true; throw s;} return f.apply(this,arguments);}; } }
</script></head><body><script>
%s
</script><script>
// take over the p5 loop: run draw() ourselves with a budget
window.__paint = (typeof draw==='function') ? draw : setup;   // many sketches paint inside setup()
window.__run=function(budget){ window.__budget=budget; window.__count=0; randomSeed(7); noiseSeed(7);
  try{ push(); window.__paint(); pop(); }catch(e){ if(!(e&&e.__stop)) window.__err=String(e); try{pop()}catch(_){} } return window.__count; };
window.__ready=false;
(function(){ let n=0; function t(){ if(++n>6){ window.__ready=true; return;} requestAnimationFrame(t);} requestAnimationFrame(t); })();
</script></body></html>"""

async def main(js, out, frames=90, seconds=6):
    from playwright.async_api import async_playwright
    code = _force_size(extract_code(Path(js).read_text()))
    # neutralise noLoop so p5 sets up but we drive draw() manually; disable p5's own draw by renaming it
    code = re.sub(r"\bfunction\s+draw\s*\(", "function draw(", code)
    prims = "[" + ",".join(f"'{p}'" for p in PRIMS) + "]"
    html = HTML % (P5, prims, code)
    tmp = Path("temp/anim"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--disable-gpu","--no-sandbox"]); pg = await b.new_page(viewport={"width":SIZE,"height":SIZE})
        await pg.set_content(html, wait_until="load"); await pg.wait_for_function("window.__ready===true", timeout=15000)
        await pg.evaluate("noLoop(); __wrap(); window.createCanvas=function(){}; window.noLoop=function(){}; window.pixelDensity=function(){};")
        total = await pg.evaluate("__run(1e9)")  # count total primitives
        err = await pg.evaluate("window.__err")
        if err: print("sketch error:", err); return
        print("primitives:", total)
        # ease-in budget schedule so the start is slow and the end fills in
        for i in range(frames):
            t = (i+1)/frames; budget = max(1, int(total * (t**1.6)))
            await pg.evaluate("window.__budget=1e9; window.__count=0; background(255)")
            await pg.evaluate(f"__run({budget})")
            await (await pg.query_selector("canvas")).screenshot(path=str(tmp/f"f{i:04d}.png"))
        await b.close()
    fps = frames/seconds
    subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",f"{fps:.3f}","-i",str(tmp/"f%04d.png"),
                    "-vf","tpad=stop_mode=clone:stop_duration=2,format=yuv420p","-c:v","libx264","-crf","18",out], check=True)
    print("wrote", out)

if __name__ == "__main__":
    a = sys.argv; asyncio.run(main(a[1], a[2], int(a[3]) if len(a)>3 else 90, float(a[4]) if len(a)>4 else 6))
