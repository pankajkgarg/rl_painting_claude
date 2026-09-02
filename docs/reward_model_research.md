# Reward-model research for p5.js painting GRPO

Research checked: 2026-09-01. Scope: local, zero-API-cost inference on eight 512×512 PNG rollouts per RL step.

## 1. Executive summary

The best answer under the current T4 constraint is **not a larger version of the present CLIP-plus-aesthetic reward**. A larger prompt-blind aesthetic regressor preserves the same exploitable objective, and HPSv3, VQAScore-XL, and the proposed small VLMs do not fit safely in the roughly 4 GB of free VRAM. The practical first choice is:

1. **T4, ~4 GB free:** HPSv2.1 plus SigLIP2-base, scored in microbatches, behind hard validity/degeneracy gates. Proposed reward: `-1` on a failed gate; otherwise `0.75 rank(HPSv2.1) + 0.25 rank(SigLIP2)`. HPSv2 supplies an actual preference signal; SigLIP2 supplies a separate alignment signal; the gate rejects blank/near-solid/blob-like exploits without turning edge count or colorfulness into another target.
2. **T4 fallback:** ImageReward plus SigLIP2-base. It is lighter than HPSv2 but older and trained on a narrower DiffusionDB-derived distribution.
3. **3090, ~10 GB free:** Qwen3-VL-2B-Instruct as a **pairwise** judge against a small hand-rated reference pool, plus HPSv2.1 and optionally SigLIP2, loaded/scored sequentially if necessary. Proposed reward: `-1` on a failed gate; otherwise `0.05 length_ok + 0.55 pairwise_win_rate + 0.30 rank(HPSv2.1) + 0.10 rank(SigLIP2)`.

This recommendation follows the closest known project precedent. Surya Narreddi's p5.brush project changed its reward to **0.05 compile-and-uses-brush + 0.05 length + 0.30 HPSv3 + 0.60 pairwise VLM wins against two references** after its earlier scalar rubric converged on the same flat five-petal clip-art flower. The post does **not** name the judge model or publish its full prompt/code, so those details cannot be reproduced exactly from the post ([Surya Narreddi, “Training AI to Paint with Code”](https://surya.website/rling-qwen-to-paint-with-code)).

### Evidence and memory conventions

- **Disk** below means the one weight representation needed at runtime, not the total size of a Hugging Face repository containing duplicate PyTorch/Safetensors formats, test images, or old checkpoints.
- Model-card parameter counts and repository file sizes are reported as published. A count inferred from weight bytes is labeled **derived**, never exact.
- I found no maintainer-published peak-VRAM or peak-RAM measurements for these exact checkpoints, 512×512 source images, and an eight-rollout step. Every footprint range labeled **planning estimate** is therefore not a benchmark. It is a conservative range derived from stored weight bytes plus inference activations, framework state, and allocator overhead. The range assumes inference mode, no gradients, and microbatch 1–2 unless stated otherwise. Measure `torch.cuda.max_memory_allocated()` in the actual training process before committing to a configuration.
- “Eight images per step” does not require all eight to reside in one scorer forward pass. Microbatching them serially keeps weight memory fixed and is the appropriate comparison for a 4 GB residual budget.

| Candidate | One-format weight size | Planning inference footprint | ~4 GB free T4 | ~10 GB free 3090 |
|---|---:|---:|---:|---:|
| HPSv2.1 | 1.97 GB FP16 ([files](https://huggingface.co/xswu/HPSv2/tree/main)) | ~2.3–3.2 GB GPU | Yes, microbatch | Yes |
| HPSv3 | 16.6 GB ([files](https://huggingface.co/MizzenAI/HPSv3/tree/main)) | ~18–22 GB GPU | No | No |
| PickScore | 3.94 GB FP32; ~1.97 GB after FP16 load ([files](https://huggingface.co/yuvalkirstain/PickScore_v1/tree/main)) | ~2.3–3.2 GB GPU | Yes, microbatch | Yes |
| ImageReward | 1.79 GB FP32 ([files](https://huggingface.co/zai-org/ImageReward/tree/main)) | ~2.2–3.0 GB GPU | Yes | Yes |
| LAION aesthetic ViT-L/14 | 1.71 GB FP32 CLIP + 3.54 MB head ([CLIP](https://huggingface.co/openai/clip-vit-large-patch14/tree/main), [head](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth)) | ~1.0–1.4 GB GPU in FP16 | Yes | Yes |
| CLIP ViT-L/14 | 1.71 GB FP32 ([files](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)) | ~1.0–1.4 GB GPU in FP16 | Yes | Yes |
| SigLIP2-base/224 | 1.50 GB FP32 ([files](https://huggingface.co/google/siglip2-base-patch16-224/tree/main)) | ~0.9–1.3 GB GPU in FP16 | Yes | Yes |
| VQAScore CLIP-FlanT5-XL | 6.33 GB BF16 ([files](https://huggingface.co/zhiqiulin/clip-flant5-xl/tree/main)) | ~7–10 GB GPU | No | Borderline |
| Qwen3-VL-2B-Instruct | 4.26 GB BF16 ([files](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main)) | ~5–7 GB GPU | No | Yes |

The disk facts in this summary come from the corresponding cited repository trees in the sections below. The footprint ranges are estimates, not sourced measurements.

## 2. Candidate evaluations

### 2.1 HPSv2 and HPSv3 (Human Preference Score)

#### HPSv2.1

**Identity, size, license, footprint**

- HF repo: [`xswu/HPSv2`](https://huggingface.co/xswu/HPSv2). The current compressed v2.1 checkpoint is `HPS_v2.1_compressed.pt`, **1.97 GB FP16**; the 12 GB repo total also contains an 8.06 GB old FP32 checkpoint and the old compressed v2 checkpoint ([file tree](https://huggingface.co/xswu/HPSv2/tree/main)).
- Architecture/parameters: the package uses LAION OpenCLIP ViT-H/14; the backbone repo reports **1.0B parameters**, while LAION's architecture breakdown gives 354.03M text plus 632.08M image parameters, about **986.11M** total ([backbone card](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K), [LAION Large OpenCLIP report](https://laion.ai/blog/large-openclip/)).
- License: **Apache-2.0** for HPSv2 code/checkpoint ([HF metadata](https://huggingface.co/xswu/HPSv2), [repository license](https://github.com/tgxs002/HPSv2/blob/master/LICENSE)). The underlying LAION CLIP-H repo is tagged **MIT** ([backbone card](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
- Planning estimate, not a published benchmark: **~2.3–3.2 GB VRAM** in FP16 for microbatch 1–2, or roughly **2.5–4 GB system RAM** if the compressed FP16 tensors remain FP16 on CPU; an FP32 CPU load is more plausibly **~4.5–6 GB RAM**. These ranges are derived from the published **1.97 GB FP16** checkpoint, with runtime overhead added ([file tree](https://huggingface.co/xswu/HPSv2/tree/main)). HPSv2 and SigLIP2 together are a tight fit in 4 GB; they should be verified in-process and, if needed, run sequentially rather than kept live together.

Install:

```bash
pip install hpsv2
```

Minimal scalar score:

```python
import hpsv2

def score_hpsv21(image_path: str, prompt: str) -> float:
    values = hpsv2.score(image_path, prompt, hps_version="v2.1")
    return float(values[0])
```

This is the official API. HPS's own documentation says comparisons are meaningful for images generated from the **same prompt**, and v2.0 values should not be compared numerically with v2.1 values ([official README](https://github.com/tgxs002/HPSv2#image-comparison)). That makes within-GRPO-group ranks a better input than globally interpreting the raw value.

**Documented failure modes / reward-hacking evidence**

- A controlled reward-model study reports that HPSv2 favors **bright lighting, complex/richly detailed composition, and vivid colors**. Optimizing BrushNet against it produced excessively bright, over-intricate, unnaturally vivid outputs; the effect depended on the base model ([ICLR 2026 paper](https://openreview.net/pdf/dfed8da21f4fcec96bf2c1c5396e02b983ba4aef.pdf)). This is directly relevant to watercolor, where saturation and decorative detail are cheap p5.js exploits.
- Direct reward fine-tuning work observed reward hacking as loss of diversity/collapse to a high-reward image and found HPSv2 generally encouraged more colorful but less photorealistic generations; optimizing one preference reward did not reliably generalize to another ([DRaFT, ICLR 2024](https://openreview.net/pdf?id=1vmSEVL19f)).
- HPSv3's authors show HPSv2-optimized examples containing meaningless accessories, unprompted objects, and decorative light spots, while presenting HPSv3 as less susceptible—not immune—to those failures ([HPSv3 supplementary material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Ma_HPSv3_Towards_Wide-Spectrum_ICCV_2025_supplemental.pdf)).
- A broader 2026 study finds that aesthetic/preference rewards and prompt-image consistency rewards can both produce artifact-prone high-reward images; ensembling only partially mitigates the problem ([Understanding Reward Hacking in Text-to-Image RL](https://arxiv.org/abs/2601.03468)). Thus HPSv2 should be one signal plus a veto/red-team process, not the sole objective.

**Assessment:** best available preference model that fits the 4 GB residual GPU budget, but its known color/detail bias means it cannot replace the current composite as a single unrestricted scalar.

#### HPSv3

**Identity, size, license, footprint**

- HF repo: [`MizzenAI/HPSv3`](https://huggingface.co/MizzenAI/HPSv3); its single `HPSv3.safetensors` checkpoint is **16.6 GB** ([file tree](https://huggingface.co/MizzenAI/HPSv3/tree/main)).
- Parameters: the maintainers describe it as built on Qwen2-VL and publish a training configuration using `Qwen/Qwen2-VL-7B-Instruct`; they call the recipe “Train with 7B model” ([official model card](https://huggingface.co/MizzenAI/HPSv3)). The exact full reward-model parameter count is **not published**. Dividing the [16.6 GB checkpoint](https://huggingface.co/MizzenAI/HPSv3/tree/main) by two bytes gives about 8.3B stored 16-bit values, but that is only a byte-derived upper-level sanity check, not a verified parameter count.
- License: HF weights are tagged **Apache-2.0** ([model card](https://huggingface.co/MizzenAI/HPSv3)); the source-code repository carries an **MIT** license ([GitHub license](https://github.com/MizzenAI/HPSv3/blob/main/LICENSE)).
- Planning estimate, not a published benchmark: at least the **16.6 GB weight payload**, and roughly **18–22 GB VRAM** or **18–24 GB RAM** after runtime/activation overhead for small-batch inference ([file tree](https://huggingface.co/MizzenAI/HPSv3/tree/main)). It cannot run in 4 GB or 10 GB free VRAM without an unverified third-party quantization/offload path, and even the checkpoint alone exceeds a 15 GB T4.

Install:

```bash
pip install hpsv3
# Optional, only on a compatible CUDA/PyTorch stack:
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Minimal scalar score:

```python
from hpsv3 import HPSv3RewardInferencer

judge = HPSv3RewardInferencer(device="cuda")

def score_hpsv3(image_path: str, prompt: str) -> float:
    # Keywords avoid an argument-order inconsistency in examples in the repo.
    out = judge.reward(prompts=[prompt], image_paths=[image_path])
    return float(out[0, 0].item())  # mu; column 1 is sigma
```

The current implementation's actual signature is `reward(self, prompts, image_paths)`, although one README example passes positional arguments in the opposite order; keyword arguments above follow the source and avoid that trap ([inference source](https://raw.githubusercontent.com/MizzenAI/HPSv3/main/hpsv3/inference.py)).

**Documented failure modes / reward-hacking evidence**

- The HPSv3 paper/supplement documents **less** obvious reward hacking than HPSv2, not a proof of robustness. It was evaluated primarily on outputs of image generators and on Stable Diffusion reward optimization, not adversarial p5.js programs ([paper](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html), [supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Ma_HPSv3_Towards_Wide-Spectrum_ICCV_2025_supplemental.pdf)). I found no primary source establishing the often-repeated claim that HPSv3 specifically over-rewards saturated/high-contrast color; that behavior is documented for **HPSv2**, so it should not be attributed to v3 without a project-specific test.
- A 2026 follow-on identifies a distribution-shift limitation of HPSv3: reward models trained on pre-annotated outputs of earlier T2I systems do not track the changing quality distribution created by newer generators and iterative RL ([HPSv3++ abstract](https://arxiv.org/abs/2606.14657)). A p5.js renderer is an even larger domain shift.
- In Surya's p5.brush experiment, the old nine-signal composite converged to a flat five-petal clip-art flower while reward increased. HPSv3 was the only component retaining real variance but had weight 0.10; therefore this incident demonstrates failure of the **composite**, not isolated proof that HPSv3 caused the flower hack. Raising HPSv3 to 0.30 and making pairwise judgment 0.60 improved the next run ([project post](https://surya.website/rling-qwen-to-paint-with-code)).
- As with other preference rewards, common artifact exploitation and mode loss remain plausible under direct maximization; preference/alignment ensembles have been shown to mitigate but not remove this class of failure ([Understanding Reward Hacking in Text-to-Image RL](https://arxiv.org/abs/2601.03468), [reward-diversity tradeoff analysis](https://arxiv.org/abs/2409.06493)).

**Assessment:** the strongest candidate conceptually and the closest to the successful precedent, but completely outside both stated residual-VRAM budgets. Use only if hosted on a separate ≥24 GB device; do not plan around it here.

### 2.2 PickScore

**Identity, size, license, footprint**

- HF model: [`yuvalkirstain/PickScore_v1`](https://huggingface.co/yuvalkirstain/PickScore_v1). Processor/backbone: [`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K).
- Parameters: **1.0B**, reported by the PickScore model card; it is CLIP-H fine-tuned on Pick-a-Pic ([model card](https://huggingface.co/yuvalkirstain/PickScore_v1)).
- Disk: one FP32 format is **3.94 GB**. The 7.89 GB repository contains the same weights in Safetensors and PyTorch formats, not two models ([file tree](https://huggingface.co/yuvalkirstain/PickScore_v1/tree/main)). Loading/casting to FP16 makes raw weight residency about **1.97 GB derived**.
- License: the weight model card has missing YAML metadata and **declares no weight license** ([model card warning](https://huggingface.co/yuvalkirstain/PickScore_v1)). The accompanying source repository is **MIT** ([code license](https://github.com/yuvalkirstain/PickScore/blob/main/LICENSE)), and the upstream LAION CLIP-H repo is tagged MIT ([backbone card](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)); neither fact supplies an explicit license for the fine-tuned PickScore weights. Treat commercial/legal use as unresolved.
- Planning estimate, not a published benchmark: **~2.3–3.2 GB VRAM** with FP16 weights and microbatch 1–2; **~4.5–6 GB RAM** in the checkpoint's native FP32, or ~2.5–4 GB if explicitly retained as FP16 on CPU. The estimate is derived from the published 3.94 GB FP32 file / 1.97 GB FP16 weight residency ([file tree](https://huggingface.co/yuvalkirstain/PickScore_v1/tree/main)).

Install:

```bash
pip install torch transformers pillow
```

Minimal scalar score (use the raw scaled similarity; a softmax over one image would always equal 1):

```python
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

device = "cuda"
processor = AutoProcessor.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)
model = AutoModel.from_pretrained(
    "yuvalkirstain/PickScore_v1", dtype=torch.float16
).eval().to(device)

def score_pickscore(image_path: str, prompt: str) -> float:
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=[image], return_tensors="pt").to(device)
    text_inputs = processor(
        text=[prompt], padding=True, truncation=True, max_length=77,
        return_tensors="pt"
    ).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        image_emb = F.normalize(model.get_image_features(**image_inputs), dim=-1)
        text_emb = F.normalize(model.get_text_features(**text_inputs), dim=-1)
        value = model.logit_scale.exp() * (text_emb @ image_emb.T)
    return float(value[0, 0].item())
```

This is the scoring operation published in the model card ([official example](https://huggingface.co/yuvalkirstain/PickScore_v1)).

**Documented failure modes / reward-hacking evidence**

- In the same controlled study that exposed HPSv2's bias, PickScore preferred **dim lighting, simple compositions with few details, and muted colors**—almost the opposite direction. The authors show that these biases propagate differently depending on the optimized base model ([ICLR 2026 paper](https://openreview.net/pdf/dfed8da21f4fcec96bf2c1c5396e02b983ba4aef.pdf)). In this project it could reward a restrained gray-disc composition for exactly the wrong reason.
- Preference rewards including PickScore can give lower scores to visually rich, high-aesthetic images when strict text alignment dominates their learned preference ([Enhancing Reward Models Beyond Text-Image Alignment, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Ba_Enhancing_Reward_Models_for_High-quality_Image_Generation_Beyond_Text-Image_Alignment_ICCV_2025_paper.pdf)).
- Direct reward fine-tuning can overfit PickScore/HPSv2 and lose diversity; cross-reward generalization is imperfect ([DRaFT, ICLR 2024](https://openreview.net/pdf?id=1vmSEVL19f)).
- The broad reward-hacking study finds artifact-prone optima across preference and alignment rewards and only partial protection from ensembles ([Understanding Reward Hacking in Text-to-Image RL](https://arxiv.org/abs/2601.03468)).

**Assessment:** technically feasible on the T4, but it shares the same 1B CLIP-H scale as HPSv2, has a documented simplicity/dimness bias that is especially concerning here, and has an undeclared checkpoint license. Rank below HPSv2 and ImageReward.

### 2.3 ImageReward

**Identity, size, license, footprint**

- Canonical HF ID is now [`zai-org/ImageReward`](https://huggingface.co/zai-org/ImageReward); the historical `THUDM/ImageReward` organization/repository redirects to Z.ai. The package model name is `ImageReward-v1.0`.
- Parameters: the canonical authors do **not publish an exact parameter count**. The FP32 checkpoint is **1.79 GB**, implying about 0.45B stored 32-bit values if it contains only parameters; a noncanonical Transformers conversion reports **0.4B**, which is corroboration rather than an official count ([canonical file tree](https://huggingface.co/zai-org/ImageReward/tree/main), [conversion card](https://huggingface.co/RE-N-Y/ImageReward)).
- Disk: `ImageReward.pt` is **1.79 GB**; the 5.44 GB repository total includes a 1.18 GB test-image archive and other assets ([file tree](https://huggingface.co/zai-org/ImageReward/tree/main)).
- License: **Apache-2.0** for code and model ([HF metadata](https://huggingface.co/zai-org/ImageReward), [repository license](https://github.com/zai-org/ImageReward/blob/main/LICENSE)).
- Planning estimate, not a published benchmark: **~2.2–3.0 GB VRAM** in the package's native FP32 path for microbatch 1–2, and **~2.2–3.5 GB system RAM**. The range is derived from the published 1.79 GB FP32 checkpoint plus working state ([file tree](https://huggingface.co/zai-org/ImageReward/tree/main)). An FP16 conversion may be smaller, but the canonical package does not document it as its standard path, so no lower number is claimed here.

Install:

```bash
pip install image-reward
```

Minimal scalar score:

```python
import ImageReward as RM

model = RM.load("ImageReward-v1.0", device="cuda")

def score_image_reward(image_path: str, prompt: str) -> float:
    return float(model.score(prompt, image_path))
```

The package and `score(prompt, image)` interface are documented by the authors ([official quick start](https://github.com/zai-org/ImageReward#imagereward)).

**Documented failure modes / reward-hacking evidence**

- Domain shift is material: ImageReward was trained on **137k expert comparison pairs** built from prompts and generated images from DiffusionDB ([official model card](https://huggingface.co/zai-org/ImageReward), [paper](https://arxiv.org/abs/2304.05977)). A 512×512 p5.js sketch is not represented by that stated source distribution. This is a documented training limitation, not itself an observed hack.
- ImageReward and other older reward models may under-score rich detail/high aesthetics because learned text-image alignment dominates the quality signal ([ICCV 2025 reward-model analysis](https://openaccess.thecvf.com/content/ICCV2025/papers/Ba_Enhancing_Reward_Models_for_High-quality_Image_Generation_Beyond_Text-Image_Alignment_ICCV_2025_paper.pdf)).
- MJ-Bench found that smaller scoring models can outperform open VLMs on some alignment/quality feedback, but no single multimodal judge is reliable across alignment, quality, safety, and bias; this argues for scoped use rather than treating ImageReward as ground truth ([MJ-Bench](https://arxiv.org/abs/2407.04842)).
- Artifact-prone high-reward images are a common failure across aesthetic/human-preference and alignment rewards, and reward ensembling only partly fixes them ([Understanding Reward Hacking in Text-to-Image RL](https://arxiv.org/abs/2601.03468)).

**Assessment:** a good lightweight baseline and the simplest T4 fallback. It is preferable to the current prompt-blind aesthetic head, but HPSv2.1 is newer and has a larger documented preference dataset. Test both on the project's blob-vs-painting red-team pairs before choosing.

### 2.4 LAION improved aesthetic predictor on CLIP ViT-L/14

**Identity, size, license, footprint**

- There is **no official Hugging Face repo ID for the aesthetic head**. The authoritative artifact is the GitHub repository [`christophschuhmann/improved-aesthetic-predictor`](https://github.com/christophschuhmann/improved-aesthetic-predictor), checkpoint `sac+logos+ava1-l14-linearMSE.pth`. The CLIP backbone can be loaded as HF ID [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14).
- Parameters: the backbone card reports **0.4B parameters**. The published MLP architecture has **927,969 parameters derived** from its layer shapes (768→1024→128→64→16→1), so the head adds under one million parameters ([CLIP card](https://huggingface.co/openai/clip-vit-large-patch14), [published inference code](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py)).
- Disk: one CLIP FP32 Safetensors file is **1.71 GB**; the aesthetic head is **3.54 MB** ([CLIP file tree](https://huggingface.co/openai/clip-vit-large-patch14/tree/main), [head file](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth)).
- License: the aesthetic-predictor repository/head is **Apache-2.0** ([repository license](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/LICENSE)). OpenAI's CLIP source code is **MIT** ([CLIP code license](https://github.com/openai/CLIP/blob/main/LICENSE)); the HF weight repository itself has no license tag, so the code license should not be silently presented as an explicit weight license ([HF model card](https://huggingface.co/openai/clip-vit-large-patch14)).
- Planning estimate, not a published benchmark: **~1.0–1.4 GB VRAM** when the CLIP backbone is FP16 and the small MLP is FP32, or **~2.0–2.7 GB RAM** with the backbone in FP32. The estimate is derived from the 1.71 GB FP32 backbone and 3.54 MB head ([files](https://huggingface.co/openai/clip-vit-large-patch14/tree/main), [head](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth)).

Install/download:

```bash
pip install torch torchvision pillow numpy \
  git+https://github.com/openai/CLIP.git
curl -L \
  'https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth' \
  -o sac+logos+ava1-l14-linearMSE.pth
```

Minimal scalar score:

```python
from PIL import Image
import clip
import torch
from torch import nn

device = "cuda"
head = nn.Sequential(
    nn.Linear(768, 1024), nn.Dropout(0.2),
    nn.Linear(1024, 128), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.Dropout(0.1),
    nn.Linear(64, 16),
    nn.Linear(16, 1),
)
head.load_state_dict(torch.load(
    "sac+logos+ava1-l14-linearMSE.pth", map_location="cpu", weights_only=True
))
head.eval().to(device)
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()

def score_laion_aesthetic_l14(image_path: str, prompt: str) -> float:
    del prompt  # The predictor is explicitly prompt-blind.
    pixels = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = clip_model.encode_image(pixels).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return float(head(emb)[0, 0].item())
```

The layer sequence and normalized CLIP embedding follow the published inference script ([source](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py)).

**Documented failure modes / reward-hacking evidence**

- The authors define the target as “how much people like on average an image”; it receives only a CLIP image embedding and **does not receive the prompt** ([official repository](https://github.com/christophschuhmann/improved-aesthetic-predictor)). Therefore it cannot distinguish “a faithful painterly hibiscus” from a generic high-scoring image that ignores the requested object. That is an architectural fact, not speculation.
- Krea reports that LAION Aesthetics is highly biased toward **women, blurry backgrounds, overly soft textures, and bright images**, and warns that relying on it adds those biases to a generator's priors ([Krea technical report](https://www.krea.ai/blog/flux-krea-open-source-release)). The same report describes downstream preference-data failures including symmetric/simple compositions, blur/softness, palette collapse, and regression to a generic “AI look.”
- Direct reward fine-tuning against the LAION aesthetic classifier has produced mode collapse/high-reward image repetition; the same work explicitly labels this reward hacking ([DRaFT, ICLR 2024](https://openreview.net/pdf/4f6a4d187763cd647549b92a33f6f1c84f23bdec.pdf)).
- Optimization against a fixed scalar reward has a fundamental reward/diversity tradeoff and can inevitably overfit without regularization ([Elucidating Optimal Reward-Diversity Tradeoffs](https://arxiv.org/abs/2409.06493)).

**Assessment:** moving from CLIP B/32 to L/14 improves feature capacity but does not repair the objective. It remains a prompt-blind average-taste regressor with documented blur/brightness/composition biases. Use at most as a low-weight diagnostic; do not make it the replacement reward.

### 2.5 CLIP ViT-L/14 or SigLIP2 for text-image alignment

These are alignment encoders, not preference/aesthetic judges. They are useful as a minority term or gate, not as the main answer to “real painterly flower versus pink blob.”

#### CLIP ViT-L/14

**Identity, size, license, footprint**

- HF ID: [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14).
- Size: **0.4B parameters** and **1.71 GB** for one FP32 Safetensors representation ([model card](https://huggingface.co/openai/clip-vit-large-patch14), [file tree](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)).
- License: OpenAI CLIP code is **MIT** ([GitHub license](https://github.com/openai/CLIP/blob/main/LICENSE)); the HF weight repo has no explicit license tag ([model card](https://huggingface.co/openai/clip-vit-large-patch14)).
- Planning estimate, not a published benchmark: **~1.0–1.4 GB VRAM** in FP16 or **~2.0–2.7 GB RAM** in FP32, derived from the 1.71 GB FP32 checkpoint ([file tree](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)).

Install:

```bash
pip install torch transformers pillow
```

Minimal scalar score:

```python
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

device = "cuda"
model_id = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id, dtype=torch.float16).eval().to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def score_clip_l14(image_path: str, prompt: str) -> float:
    inputs = processor(
        text=[prompt], images=Image.open(image_path).convert("RGB"),
        return_tensors="pt", padding=True
    ).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # Raw one-pair logit. Do not softmax a single candidate.
        value = model(**inputs).logits_per_image[0, 0]
    return float(value.item())
```

#### SigLIP2-base/224

**Identity, size, license, footprint**

- HF ID: [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224).
- Size: **0.4B parameters** and a **1.50 GB FP32** Safetensors file (1.54 GB repo including tokenizer) ([model card](https://huggingface.co/google/siglip2-base-patch16-224), [file tree](https://huggingface.co/google/siglip2-base-patch16-224/tree/main)).
- License: **Apache-2.0** ([HF model card](https://huggingface.co/google/siglip2-base-patch16-224)).
- Planning estimate, not a published benchmark: **~0.9–1.3 GB VRAM** in FP16 or **~1.8–2.5 GB RAM** in FP32, derived from the 1.50 GB FP32 checkpoint ([file tree](https://huggingface.co/google/siglip2-base-patch16-224/tree/main)).

Install:

```bash
pip install torch transformers pillow sentencepiece
```

Minimal scalar score:

```python
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

device = "cuda"
model_id = "google/siglip2-base-patch16-224"
model = AutoModel.from_pretrained(model_id, dtype=torch.float16).eval().to(device)
processor = AutoProcessor.from_pretrained(model_id)

def score_siglip2(image_path: str, prompt: str) -> float:
    inputs = processor(
        text=[prompt], images=[Image.open(image_path).convert("RGB")],
        padding="max_length", truncation=True, max_length=64, return_tensors="pt"
    ).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        probability = torch.sigmoid(model(**inputs).logits_per_image[0, 0])
    return float(probability.item())
```

The official card describes SigLIP2 as an image-text retrieval / zero-shot-classification encoder, not an aesthetic model ([model card](https://huggingface.co/google/siglip2-base-patch16-224)).

**Documented failure modes / reward-hacking evidence**

- CLIP's own model card says it struggles with **fine-grained classification and counting** and that results/biases depend materially on class design ([CLIP model card](https://huggingface.co/openai/clip-vit-large-patch14)). Those are exactly the kinds of compositional distinctions a coarse flower silhouette can satisfy without being a good painting.
- OpenAI documented **typographic attacks**: rendered words can activate CLIP concepts and override visual evidence, even when the text conflicts absurdly with the image ([OpenAI multimodal-neuron study](https://openai.com/index/multimodal-neurons/)). A code-generating policy can trivially learn to draw prompt words, so p5.js `text()`/font/glyph use should be disallowed or independently penalized if CLIP/SigLIP is a reward.
- For SigLIP2 specifically, I found no primary study documenting a p5.js or direct-RL saturation/color hack. The verified limitation is scope: its card validates retrieval/classification, not painterly taste ([SigLIP2 card](https://huggingface.co/google/siglip2-base-patch16-224)). It should therefore not inherit claims established only for CLIP.
- At the category level, the 2026 reward-hacking analysis shows that **prompt-image consistency rewards also produce artifact-prone optima**, and combining them with preference rewards only partially mitigates hacking ([Understanding Reward Hacking in Text-to-Image RL](https://arxiv.org/abs/2601.03468)).

**Assessment:** choose SigLIP2-base as the auxiliary alignment term: it has a clear Apache license and lower weight residency. Neither alignment encoder judges brushwork, depth, composition quality, or painterliness.

### 2.6 VQAScore

The original, reproducible VQAScore is based on CLIP-FlanT5. The current `t2v_metrics` 3.1 package also exposes VQAScore through Qwen3-VL and other VLMs, but that overlaps the small-VLM candidate in §2.7; the resource accounting here uses the original open CLIP-FlanT5 checkpoints.

**Identity, size, license, footprint**

- HF IDs: [`zhiqiulin/clip-flant5-xl`](https://huggingface.co/zhiqiulin/clip-flant5-xl) and [`zhiqiulin/clip-flant5-xxl`](https://huggingface.co/zhiqiulin/clip-flant5-xxl). The official code/model registry is [`linzhiqiu/t2v_metrics`](https://github.com/linzhiqiu/t2v_metrics).
- Parameters: XL uses a FLAN-T5-XL **3B** language backbone; XXL uses FLAN-T5-XXL **11B** ([FLAN-T5-XL card](https://huggingface.co/google/flan-t5-xl), [FLAN-T5-XXL card](https://huggingface.co/google/flan-t5-xxl)). The multimodal checkpoints' exact full parameter counts are not explicitly published. The [XL 6.33 GB BF16 payload](https://huggingface.co/zhiqiulin/clip-flant5-xl/tree/main) corresponds to about **3.16B stored 16-bit values derived**; the [XXL 22.9 GB payload](https://huggingface.co/zhiqiulin/clip-flant5-xxl/tree/main) corresponds to about 11.45B stored values derived.
- Disk: **6.33 GB BF16** for CLIP-FlanT5-XL and **22.9 GB BF16** for XXL ([XL file tree](https://huggingface.co/zhiqiulin/clip-flant5-xl/tree/main), [XXL file tree](https://huggingface.co/zhiqiulin/clip-flant5-xxl/tree/main)).
- License: the `t2v_metrics` code and both model cards are **Apache-2.0** ([code license](https://github.com/linzhiqiu/t2v_metrics/blob/main/LICENSE), [XL card](https://huggingface.co/zhiqiulin/clip-flant5-xl), [XXL card](https://huggingface.co/zhiqiulin/clip-flant5-xxl)).
- Planning estimates, not published benchmarks: XL is roughly **~7–10 GB VRAM** or **~7–10 GB RAM** at BF16 for one-image/microbatch inference; XXL is roughly **~25–32 GB VRAM/RAM**. These derive from the 6.33/22.9 GB checkpoints plus generation/runtime overhead ([XL files](https://huggingface.co/zhiqiulin/clip-flant5-xl/tree/main), [XXL files](https://huggingface.co/zhiqiulin/clip-flant5-xxl/tree/main)). The authors' legacy documentation recommends the largest model only with a 40 GB GPU and offers XL for limited memory ([v3.0 README](https://github.com/linzhiqiu/t2v_metrics/blob/main/V_3.0_README.md)). XL does not fit 4 GB; it is only borderline in 10 GB and would consume almost the entire scorer allowance.

Install the paper-reproducible legacy implementation:

```bash
pip install t2v-metrics==3.0
```

Minimal scalar score:

```python
import t2v_metrics

model = t2v_metrics.VQAScore(model="clip-flant5-xl")

def score_vqa(image_path: str, prompt: str) -> float:
    value = model(images=[image_path], texts=[prompt])
    return float(value[0, 0].item())
```

The original score is the model probability of answering “Yes” to a question equivalent to “Does this figure show {prompt}?”; the authors note that template changes affect results and discourage casual modification when reproducing the metric ([legacy documentation](https://github.com/linzhiqiu/t2v_metrics/blob/main/V_3.0_README.md)).

**Documented failure modes / reward-hacking evidence**

- GenAI-Bench reports that VQAScore substantially improves compositional alignment ranking over CLIPScore, PickScore, HPSv2, and ImageReward, but calls out remaining weaknesses on **fine-grained visual details** ([GenAI-Bench](https://arxiv.org/abs/2406.13743)).
- The paper's error analysis identifies difficulty with **larger counts, small/fine details, and ambiguous language/conventions** such as “two shoes” versus two pairs and viewpoint-dependent left/right ([CVPR 2024 workshop paper](https://openaccess.thecvf.com/content/CVPR2024W/EvGenFM/papers/Li_Evaluating_and_Improving_Compositional_Text-to-Visual_Generation_CVPRW_2024_paper.pdf)).
- VQAScore answers a prompt-presence question. It is therefore principally an alignment score, not a direct measure of brushwork, composition, depth, or aesthetic quality; that definition is stated by its authors ([official repository](https://github.com/linzhiqiu/t2v_metrics), [VQAScore paper](https://arxiv.org/abs/2404.01291)).
- Prompt-image consistency rewards as a class remain hackable through artifact-prone images under RL optimization ([Understanding Reward Hacking in Text-to-Image RL](https://arxiv.org/abs/2601.03468)). I found no published adversarial p5.js optimization audit for CLIP-FlanT5 itself.

**Assessment:** strong alignment metric, but the XL checkpoint is too large for 4 GB and gives less direct aesthetic leverage than a 2B pairwise VLM in the 10 GB scenario. Do not pay its memory cost merely to replace CLIP on simple flower prompts.

### 2.7 Small open VLM as a pairwise or scalar judge

**Identity, size, license, footprint**

Preferred checkpoint:

- [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct): **2B parameters**, **4.26 GB BF16 weights** (4.27 GB repository), **Apache-2.0** ([model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [file tree](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main)).
- Planning estimate, not a published benchmark: **~5–7 GB VRAM** or **~5–8 GB RAM** for one 512×512 image, a short rubric, and a few generated tokens. This is derived from the published 4.26 GB BF16 checkpoint plus vision tokens, KV cache, and framework overhead ([file tree](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main)). It does not fit the 4 GB residual budget but does fit 10 GB with conservative context/image limits.

Older alternatives:

- [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) is branded 3B, while HF reports **4B actual parameters** and BF16; its weights occupy **7.52 GB** ([model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [file tree](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main)). Planning estimate: **~8.5–11 GB VRAM**, so it is worse than Qwen3-VL-2B for this budget.
- [`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ) stores roughly **3.4 GB** of 4-bit weights ([file tree](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ/tree/main)). Planning estimate: **~4.5–6 GB VRAM** after vision/runtime overhead; the weight file fitting under 4 GB does not mean the complete model fits.
- License caveat: the official Qwen2.5-VL-3B checkpoint uses the **Qwen Research License**, which permits **non-commercial purposes only** and requires a separate commercial license ([raw license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE)). This is another reason to prefer Apache-licensed Qwen3-VL-2B.

Install Qwen3-VL using the maintainers' source-install guidance:

```bash
pip install git+https://github.com/huggingface/transformers accelerate pillow
```

Minimal scalar score. This is deliberately deterministic and short, but pairwise judgment is recommended for training:

```python
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_id = "Qwen/Qwen3-VL-2B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

RUBRIC = """Judge this rendered p5.js sketch against the requested prompt: {prompt}
Score overall visual success from 0 to 10. Reward prompt-faithful recognizability,
painterly brushwork, composition, depth, and intentional detail. Penalize flat clip-art,
simple blobs/discs, text drawn into the image, artifacts, and irrelevant decoration.
Return only one integer from 0 to 10."""

def score_qwen3_vl(image_path: str, prompt: str) -> float:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": Path(image_path).resolve().as_uri()},
            {"type": "text", "text": RUBRIC.format(prompt=prompt)},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    answer = processor.decode(
        output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    return float(answer) / 10.0
```

The load/chat-template pattern is from the official Qwen3-VL model card ([quick start](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)). Production code needs strict output validation and a fallback for malformed responses.

**Documented failure modes / reward-hacking evidence**

- In the closest project, absolute VLM scores from 0–10 were **compressed near zero**; asking which image was better against references opened useful dynamic range ([Surya post](https://surya.website/rling-qwen-to-paint-with-code)). This is direct evidence to prefer pairwise wins over the illustrative scalar snippet above.
- Multi-image VLMs have pronounced **position bias**: moving the same evidence between image positions changes predictions. The CVPR 2025 study finds open models often reason better about later images than earlier/middle images ([position-bias paper](https://arxiv.org/abs/2503.13792)). Randomize A/B order and average or require consistency over both orders.
- VL-RewardBench reports that even GPT-4o achieved only 65.4% on its difficult vision-language reward benchmark, while Qwen2-VL-72B struggled to exceed chance; failures were mainly basic visual perception rather than reasoning ([VL-RewardBench](https://arxiv.org/abs/2411.17451)). A generic 2B model should not be presumed to be a calibrated aesthetic judge merely because it follows the rubric.
- MJ-Bench found that open VLMs can be worse than small dedicated scoring models for text-image alignment and image quality, though VLMs were better for safety and generation-bias judgments ([MJ-Bench](https://arxiv.org/abs/2407.04842)). This supports combining a VLM's relative holistic judgment with a dedicated preference score rather than replacing every signal.
- A controlled 2026 audit of a larger Qwen2.5-VL-7B judge found it tracked degradation magnitude reliably on only **3 of 12** tested dimensions, versus 11/12 for Gemini, with limited cross-judge agreement ([VLM-as-judge audit](https://www.utupub.fi/items/57c47809-1e08-4a5a-b368-780a22578add)). That is a different checkpoint/task and cannot be transferred as an exact accuracy number for Qwen3-VL-2B, but it is strong evidence that an in-domain sensitivity test is mandatory.
- I found no primary benchmark of the exact Qwen3-VL-2B checkpoint as a pairwise judge for p5.js paintings. All quality claims for that exact use are therefore unverified until tested on the project's own labeled adversarial pairs.

**Assessment:** best qualitative component under 10 GB, but only as a pairwise, reference-anchored judge with order swapping and an in-domain audit. It does not fit 4 GB in BF16.

## 3. Kickingkeys / Surya Narreddi project details

The requested post exists: [Surya Narreddi, “Training AI to Paint with Code”](https://surya.website/rling-qwen-to-paint-with-code), dated March 2026. The author describes training a Qwen coding model to write complete p5.brush sketches, rendering them in a sandboxed Puppeteer environment, and scoring each PNG against **two random reference paintings** from a hand-rated pool.

### Old reward and observed hack

The original reward had nine signals:

- compilation;
- actually using p5.brush rather than native p5;
- a code-length ramp targeting about 3,000 tokens;
- HPSv3;
- prompt adherence from a council of GPT-5.4 and Gemini;
- recognizability, aesthetics, technique, and depth judges.

The run plateaued around 0.65 and every rollout became the same flat five-rounded-petal clip-art flower while reward kept rising. The four quality judges plus prompt adherence correlated **0.85–0.95**. Code length supplied roughly one-third of total reward and had saturated by step 30. HPSv3 was the only signal with real variance but had weight **0.10** ([post, Reward Functions](https://surya.website/rling-qwen-to-paint-with-code)).

### Exact revised reward

The replacement was exactly:

\[
R = 0.05\,C + 0.05\,L + 0.30\,HPSv3 + 0.60\,J_{pairwise}
\]

where:

- `C` is a binary **compile-and-uses-brush** gate/component;
- `L` is a binary **length** check;
- `HPSv3` has weight **0.30**;
- `J_pairwise` is the **fraction of comparisons won** against references and has weight **0.60**.

Each rollout was compared with **two random references** from the pool. The post describes the pairwise question as: **“which of these is the better hibiscus watercolour?”** It contrasts this with an earlier 0–10 absolute judge whose scores were compressed near zero. The same base model and data reached the previous plateau three times faster and continued upward; generated code shrank from about 13,500 tokens to under 2,000 ([post](https://surya.website/rling-qwen-to-paint-with-code)).

### Reference pool

- 1,664 images were hand-rated one at a time as `love`, `okay`, or `nope`.
- The initial pairwise pool was seeded with **117 love-tier** images.
- The final pool contained **581 model-generated references**: 117 love, 266 okay, and 198 supplementary generations used to widen thin color regions.
- The reference-generation pipelines used Opus 4.6, GPT-5.4, and Gemini 3.1 Pro iterating under a VLM judge, plus a larger Gemini 3.1 Pro batch ([post, Reference Pool](https://surya.website/rling-qwen-to-paint-with-code)).

### Details the post does not disclose

The post does **not** identify the separate pairwise training judge model, checkpoint, provider, size, or inference settings. It also does not publish the complete system/user prompt template, parsing code, tie handling, A/B randomization, or score-normalization code. The named Opus/GPT/Gemini models belong to reference generation and the older prompt-adherence council; they should not be misreported as the pairwise judge. The only pairwise prompt wording shown is the single natural-language question above. No exact judge-model claim beyond that can be verified from the post.

## 4. Ranked recommendation for a 15 GB T4 with ~4 GB free

### Ranking

1. **HPSv2.1 + SigLIP2-base/224 + hard anti-exploit gates.** HPSv2 is the strongest fitting preference signal; SigLIP2 is a lower-memory, Apache-licensed alignment signal. Score the eight rollouts in microbatches of one or two. Estimated combined FP16 weight residency is about 2.72 GB (1.97 GB HPSv2 checkpoint plus ~0.75 GB derived FP16 SigLIP2 weights), leaving little runtime headroom; this is a planning calculation from the published checkpoint files, not a guarantee ([HPSv2 files](https://huggingface.co/xswu/HPSv2/tree/main), [SigLIP2 files](https://huggingface.co/google/siglip2-base-patch16-224/tree/main)). If in-process peak memory exceeds 4 GB, keep HPSv2 on GPU and run SigLIP2 sequentially/on CPU rather than removing the preference model.
2. **ImageReward + SigLIP2-base/224.** More comfortable weight footprint and an easy API, but older/narrower preference data ([ImageReward card](https://huggingface.co/zai-org/ImageReward)).
3. **HPSv2.1 alone plus gates.** Better than replacing the current reward with another aesthetic head, but less defense against prompt drift.
4. **PickScore + a gate.** Fits in FP16, but its documented dim/simple/muted bias is aligned with the observed gray-disc exploit and the fine-tuned weight license is undeclared ([bias study](https://openreview.net/pdf/dfed8da21f4fcec96bf2c1c5396e02b983ba4aef.pdf), [model card](https://huggingface.co/yuvalkirstain/PickScore_v1)).
5. **LAION aesthetic L/14 or CLIP L/14 alone.** Cheap enough, but they reproduce the wrong objective class. Use only for diagnostics or a very small auxiliary weight.
6. **Do not use here:** HPSv3, VQAScore-XL/XXL, Qwen3-VL-2B, or Qwen2.5-VL-3B. Their checkpoint/runtime footprints exceed 4 GB.

### Concrete proposed T4 reward

For the eight rollouts generated for the same prompt, define `rank8(x) = (rank(x)-1)/7`, with ties averaged, so each model contributes a comparable `[0,1]` within-group signal without pretending their raw scales are calibrated.

Define:

- `G_valid ∈ {0,1}`: compilation succeeds, a valid 512×512 render is produced, required p5.brush calls are used, and disallowed text/font/glyph APIs are absent. Rendered text is a known CLIP attack surface ([OpenAI typographic-attack analysis](https://openai.com/index/multimodal-neurons/)).
- `G_nondegenerate ∈ {0,1}`: a conservative veto trained/calibrated on project examples returns 0 for blank/near-solid frames, one-blob-on-disc cases, and gross corruption. It must be a **gate**, not a positive reward for edge density, saturation, number of shapes, or entropy; otherwise the policy will optimize that proxy too.
- `H = rank8(HPSv2.1(image, prompt))`.
- `S = rank8(SigLIP2(image, prompt))`.

Recommended starting reward:

\[
R_{T4} =
\begin{cases}
-1, & G_{valid}G_{nondegenerate}=0 \\
0.75H + 0.25S, & G_{valid}G_{nondegenerate}=1
\end{cases}
\]

Why these weights: the primary problem is visual preference/quality, so HPSv2 gets the majority. SigLIP2 supplies nonidentical prompt alignment but is capped at 0.25 because an alignment encoder cannot judge painterliness and can itself be hacked. Remove the current LAION B/32 aesthetic term rather than adding it to this formula.

Practical scoring schedule for eight images:

1. Apply the CPU-cheap validity and degeneracy gates to all eight.
2. Encode the prompt once per scorer.
3. Score valid images in HPSv2 microbatches of 1–2, then SigLIP2 microbatches of 1–2.
4. Rank within the eight-image prompt group and compute the formula.
5. Log every subreward and periodically inspect the top raw-HPS, top raw-SigLIP, and top-composite image; correlated upward scores are not evidence that human quality improved.

Before RL, build a fixed red-team set of real project renders: flat blob versus painterly flower, text-written prompt versus painted subject, saturation variants, blur/soft-focus variants, decorative-light variants, and near-duplicates. Do not train against this reward until it consistently ranks the intended winner, and keep a separate withheld set for detecting later overoptimization. This is a proposed project gate, not a published threshold.

### Two-CPU-core alternative

No primary source publishes two-core latency for these exact checkpoints, so throughput cannot be promised. The least risky CPU option is **ImageReward alone plus the deterministic gates**, scoring the eight images serially; its canonical FP32 checkpoint is 1.79 GB ([files](https://huggingface.co/zai-org/ImageReward/tree/main)). A second SigLIP2 FP32 resident model adds 1.50 GB of weights before overhead ([files](https://huggingface.co/google/siglip2-base-patch16-224/tree/main)) and is likely to make a two-core RL loop latency-bound. If CPU scoring is mandatory, benchmark one full eight-image step before selecting the composite; there is no verified latency number to report.

## 5. Ranked recommendation for a 24 GB RTX 3090 with vLLM using ~14 GB

### Ranking

1. **Qwen3-VL-2B pairwise reference judge + HPSv2.1; add SigLIP2 only if measured peak stays below 10 GB.** Qwen3-VL-2B is Apache-licensed, 4.26 GB BF16, and substantially smaller than the older nominal 3B Qwen2.5-VL checkpoint ([Qwen3 card/files](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [tree](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main)).
2. **Qwen3-VL-2B pairwise judge + HPSv2.1, without SigLIP2.** This is the safer steady-state memory configuration; the VLM rubric already sees the prompt.
3. **VQAScore-XL + HPSv2.1, scored sequentially.** Better compositional checking than CLIP but almost no VRAM slack and no direct pairwise taste anchoring.
4. **HPSv2.1 + SigLIP2 without VLM.** Same reliable lightweight fallback as the T4.
5. **Not viable:** HPSv3's 16.6 GB checkpoint or VQAScore-XXL's 22.9 GB checkpoint, before activations ([HPSv3 files](https://huggingface.co/MizzenAI/HPSv3/tree/main), [XXL files](https://huggingface.co/zhiqiulin/clip-flant5-xxl/tree/main)).

Qwen3-VL-2B + HPSv2.1 + SigLIP2 has about **6.98 GB derived weight residency** in BF16/FP16 (4.26 + 1.97 + 0.75 GB), leaving roughly 3 GB of the scorer allowance for vision activations, KV cache, framework overhead, and allocator fragmentation. The inputs are the published [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main), [HPSv2](https://huggingface.co/xswu/HPSv2/tree/main), and [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224/tree/main) files; the combined figure is plausible but not verified. Restrict the VLM to 512×512 input, a short rubric, and ≤4 output tokens; use microbatch 1–2. If vLLM has actually reserved rather than merely allocated its 14 GB budget, measure coexistence in the same process. The robust fallback is sequential residency/offload or dropping SigLIP2 first.

### Concrete proposed 3090 reward

Create a hand-rated reference pool specific to this project, with prompt/style/color strata and explicit negative examples. For each rollout, sample two references from the same prompt family or a close semantic/style bucket. For each reference, ask the VLM to compare candidate versus reference, then repeat the same comparison with image order swapped. A consistent candidate win counts 1, a consistent loss 0, and an inconsistent/order-sensitive result 0.5 or is discarded and resampled. Define `J` as the mean over the two reference comparisons. This is more expensive (up to 32 very-short judge generations for eight rollouts with two references and both orders) but remains inside the same model-memory budget; latency must be measured.

Suggested pairwise prompt template:

```text
Requested painting: {prompt}

Compare Image A and Image B as finished p5.js watercolor-style artworks.
Choose the image that better combines:
1. faithful and recognizable depiction of the requested subject;
2. painterly brushwork, color variation, layering, and depth;
3. intentional composition and useful detail;
4. absence of flat clip-art blobs, plain discs, rendered words, irrelevant
   decoration, corrupt geometry, or obvious reward-targeting tricks.

Judge relative quality, not which image is more saturated or more complex.
Return exactly A, B, or TIE.
```

Let:

- `G_valid` and `G_nondegenerate` be the same hard gates used above;
- `L_ok ∈ {0,1}` be a broad binary code-length sanity band, not a length ramp;
- `J ∈ [0,1]` be swapped-order pairwise win rate against two references;
- `H = rank8(HPSv2.1)`;
- `S = rank8(SigLIP2)`.

Recommended starting reward:

\[
R_{3090} =
\begin{cases}
-1, & G_{valid}G_{nondegenerate}=0 \\
0.05L_{ok} + 0.55J + 0.30H + 0.10S,
    & G_{valid}G_{nondegenerate}=1
\end{cases}
\]

This deliberately resembles the successful Surya recipe while substituting a model that fits: the pairwise, taste-anchored signal remains the majority; HPS supplies a stable dedicated preference feature; SigLIP is a small alignment guard; length cannot dominate. If peak memory or latency is unacceptable, drop SigLIP2 and use:

\[
R_{3090,lean} = 0.05L_{ok} + 0.65J + 0.30H
\]

with the same `-1` return when either hard gate fails.

The proposed VLM weights are not claimed to reproduce Surya's undisclosed judge. They are a budget-constrained adaptation. Before RL, require the exact pairwise pipeline—including both A/B orders—to pass the same adversarial hand-labeled set used for the T4 reward. During RL, keep a frozen human-rated canary set and monitor diversity, not only mean reward; direct optimization of fixed image rewards is known to trade reward against diversity and can collapse modes ([reward-diversity analysis](https://arxiv.org/abs/2409.06493)).

## 6. Sources

### Model repositories and official documentation

- [HPSv2 repository / usage](https://github.com/tgxs002/HPSv2) · [HPSv2 HF files](https://huggingface.co/xswu/HPSv2/tree/main) · [HPSv2 paper](https://arxiv.org/abs/2306.09341)
- [HPSv3 repository](https://github.com/MizzenAI/HPSv3) · [HPSv3 HF card](https://huggingface.co/MizzenAI/HPSv3) · [HPSv3 HF files](https://huggingface.co/MizzenAI/HPSv3/tree/main) · [HPSv3 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html)
- [PickScore HF card](https://huggingface.co/yuvalkirstain/PickScore_v1) · [PickScore files](https://huggingface.co/yuvalkirstain/PickScore_v1/tree/main) · [PickScore code](https://github.com/yuvalkirstain/PickScore)
- [ImageReward HF card](https://huggingface.co/zai-org/ImageReward) · [ImageReward files](https://huggingface.co/zai-org/ImageReward/tree/main) · [ImageReward code](https://github.com/zai-org/ImageReward) · [paper](https://arxiv.org/abs/2304.05977)
- [Improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) · [CLIP ViT-L/14 card](https://huggingface.co/openai/clip-vit-large-patch14) · [CLIP files](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
- [SigLIP2-base card](https://huggingface.co/google/siglip2-base-patch16-224) · [SigLIP2 files](https://huggingface.co/google/siglip2-base-patch16-224/tree/main) · [SigLIP2 paper](https://arxiv.org/abs/2502.14786)
- [VQAScore/t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) · [legacy v3.0 documentation](https://github.com/linzhiqiu/t2v_metrics/blob/main/V_3.0_README.md) · [CLIP-FlanT5-XL files](https://huggingface.co/zhiqiulin/clip-flant5-xl/tree/main) · [XXL files](https://huggingface.co/zhiqiulin/clip-flant5-xxl/tree/main)
- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) · [files](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main) · [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) · [Qwen2.5 3B license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE)

### Reward-hacking, bias, and judge-reliability sources

- [Surya Narreddi, “Training AI to Paint with Code”](https://surya.website/rling-qwen-to-paint-with-code)
- [Understanding Reward Hacking in Text-to-Image Reinforcement Learning](https://arxiv.org/abs/2601.03468)
- [Elucidating Optimal Reward-Diversity Tradeoffs in Text-to-Image Diffusion Models](https://arxiv.org/abs/2409.06493)
- [DRaFT: Directly Fine-tuning Diffusion Models on Differentiable Rewards](https://openreview.net/pdf?id=1vmSEVL19f)
- [ICLR 2026 study of reward-model brightness/composition/color biases](https://openreview.net/pdf/dfed8da21f4fcec96bf2c1c5396e02b983ba4aef.pdf)
- [Krea technical report on LAION Aesthetics and the “AI look”](https://www.krea.ai/blog/flux-krea-open-source-release)
- [OpenAI CLIP typographic attacks](https://openai.com/index/multimodal-neurons/)
- [GenAI-Bench / VQAScore](https://arxiv.org/abs/2406.13743)
- [MJ-Bench](https://arxiv.org/abs/2407.04842) · [VL-RewardBench](https://arxiv.org/abs/2411.17451)
- [Position bias of multi-image VLMs](https://arxiv.org/abs/2503.13792)
- [Controlled VLM-as-judge degradation audit](https://www.utupub.fi/items/57c47809-1e08-4a5a-b368-780a22578add)
