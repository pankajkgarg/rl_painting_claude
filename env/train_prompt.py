"""Prompt definition shared by train.py and eval_samples.py."""
SUBJECTS = [
    "a watercolour painting of a pink hibiscus flower",
    "a loose watercolour painting of a red poppy field under a pale sky",
    "an impressionist oil painting of a blue iris in a garden",
    "a watercolour painting of a yellow sunflower with soft green leaves",
    "a gouache painting of an orange tulip with a moody dark background",
]
SYSTEM = (
    "You are a generative artist writing p5.js sketches. Reply with ONLY one complete p5.js sketch inside a "
    "single ```javascript code block, nothing else. Rules: define function setup() and function draw(); call "
    "createCanvas(512, 512) and noLoop() in setup; draw everything inside draw(). Use only drawing primitives "
    "(background, fill, stroke, noStroke, noFill, ellipse, circle, rect, line, bezier, curveVertex, beginShape/endShape, "
    "arc, push/pop, translate, rotate, scale, random, noise, lerpColor, color, strokeWeight, blendMode). "
    "Colours are RGB 0-255 with alpha 0-255: fill(r, g, b, a). Never call colorMode, text, loadPixels, pixels, images, "
    "fetch, DOM, or infinite loops. No comments, no explanations, close the code block."
)
def make_prompt(subj):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Paint: {subj}. Technique: build the picture from at least 200 overlapping translucent shapes drawn in loops (alpha 15-90), "
            f"with random() and noise() jitter in position, size, rotation and colour so edges look like brush marks. "
            f"Paint the background first with many soft strokes, then the subject with layered petals/leaves/stems using "
            f"push/rotate/translate and bezier or curveVertex outlines. Not concentric circles. Under 80 lines, short names."}]

