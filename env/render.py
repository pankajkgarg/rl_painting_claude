"""Render p5.js sketches to PNG with headless Chromium (Playwright).

render_batch(codes) -> list of dict(ok, png_bytes|None, error, blank)
One browser, N concurrent pages, hard per-sketch timeout.
"""
import asyncio, base64, io, os, re, sys, json
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
P5 = (HERE / "p5.min.js").read_text()
SIZE = int(os.environ.get("RLPAINT_SIZE", "512"))
TIMEOUT_MS = int(os.environ.get("RLPAINT_TIMEOUT_MS", "8000"))

HTML = """<!doctype html><html><head><meta charset=utf-8>
<style>body{margin:0;background:#fff}canvas{display:block}</style>
<script>%s</script>
<script>
window.__err=null;window.__done=false;
window.addEventListener('error',e=>{window.__err=String(e.message||e.error)});
// force fixed canvas size regardless of what the sketch asks for
const _cc=window.createCanvas;
</script>
</head><body>
<script>
try{
%s
}catch(e){window.__err=String(e)}
</script>
<script>
// after p5 runs draw() for a few frames, mark done
(function(){let n=0;function tick(){n++;if(window.__err||n>12){window.__done=true;return;}requestAnimationFrame(tick)}requestAnimationFrame(tick)})();
</script>
</body></html>"""

FORBIDDEN = [
    r"\btext\s*\(", r"\btextFont\b", r"\btextSize\b", r"\bloadImage\b", r"\bcreateImg\b",
    r"\bfetch\s*\(", r"\bXMLHttpRequest\b", r"https?://", r"\bdocument\.", r"\bwindow\.",
    r"\bimport\b", r"\brequire\s*\(", r"\beval\s*\(", r"\bFunction\s*\(", r"\bloadJSON\b",
    r"\bloadStrings\b", r"\bloadFont\b", r"\bcreateCapture\b", r"\bsaveCanvas\b", r"\bsave\s*\(",
    r"\bwhile\s*\(\s*true\s*\)", r"\bnoCanvas\b",
    r"\bcolorMode\b", r"\bloadPixels\b", r"\bupdatePixels\b", r"\bpixels\s*\[", r"\bset\s*\(\s*\w+\s*,\s*\w+\s*,",  # alpha-scale trap + raw pixel hacks
]

def static_check(code: str):
    """Return None if ok, else a reason string."""
    for pat in FORBIDDEN:
        if re.search(pat, code):
            return f"forbidden: {pat}"
    if "function setup" not in code:
        return "no setup()"
    if "createCanvas" not in code:
        return "no createCanvas"
    return None

def extract_code(text: str) -> str:
    """Pull the first ```javascript / ```js fenced block; else the raw text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # strip any reasoning block
    m = re.search(r"```(?:javascript|js)?\s*\n(.*?)```", text, re.S)
    if m: return m.group(1).strip()
    m = re.search(r"```(?:javascript|js)?\s*\n(.*)$", text, re.S)  # unterminated fence (truncated)
    return (m.group(1) if m else text).strip()

def _force_size(code: str) -> str:
    # Rewrite createCanvas(...) to fixed size; keep 2D renderer.
    return re.sub(r"createCanvas\s*\([^)]*\)", f"createCanvas({SIZE},{SIZE})", code, count=1)

def blank_score(img: Image.Image) -> float:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    return float(a.std())

async def _render_one(browser, code: str):
    res = {"ok": False, "png": None, "error": None, "std": 0.0}
    reason = static_check(code)
    if reason:
        res["error"] = reason
        return res
    page = await browser.new_page(viewport={"width": SIZE, "height": SIZE})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    try:
        await page.set_content(HTML % (P5, _force_size(code)), wait_until="load")
        try:
            await page.wait_for_function("window.__done===true", timeout=TIMEOUT_MS)
        except Exception:
            res["error"] = "timeout"
            return res
        err = await page.evaluate("window.__err")
        if err or errs:
            res["error"] = (err or errs[0])[:300]
            return res
        canvas = await page.query_selector("canvas")
        if canvas is None:
            res["error"] = "no canvas"
            return res
        png = await canvas.screenshot(type="png")
        img = Image.open(io.BytesIO(png))
        res["std"] = blank_score(img)
        if res["std"] < 4.0:
            res["error"] = "blank"
            return res
        res["ok"] = True
        res["png"] = png
        return res
    except Exception as e:
        res["error"] = f"render: {e}"[:300]
        return res
    finally:
        await page.close()

async def _render_batch(codes):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
        try:
            sem = asyncio.Semaphore(int(os.environ.get("RLPAINT_CONC", "8")))
            async def guarded(c):
                async with sem:
                    return await asyncio.wait_for(_render_one(browser, c), timeout=TIMEOUT_MS/1000 + 10)
            outs = await asyncio.gather(*[guarded(c) for c in codes], return_exceptions=True)
            return [o if isinstance(o, dict) else {"ok": False, "png": None, "error": f"outer: {o}", "std": 0.0} for o in outs]
        finally:
            await browser.close()

def render_batch(codes):
    return asyncio.run(_render_batch(codes))

if __name__ == "__main__":
    # smoke: render files given on argv, write PNGs next to them
    codes = [Path(f).read_text() for f in sys.argv[1:]]
    outs = render_batch(codes)
    for f, o in zip(sys.argv[1:], outs):
        print(f, "ok" if o["ok"] else "FAIL", o["error"], round(o["std"], 1))
        if o["png"]:
            Path(f).with_suffix(".png").write_bytes(o["png"])
