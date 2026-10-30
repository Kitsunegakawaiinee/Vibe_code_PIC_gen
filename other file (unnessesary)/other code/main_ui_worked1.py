import os
import time
import random
import inspect
import threading
from datetime import datetime

import gradio as gr
from stable_diffusion_cpp import StableDiffusion

# ----------------------------- CONFIG -----------------------------
MODEL_PATH = "model/ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
VAE_PATH = ""            # optional: "model/sdxl_vae_fp16_fix.safetensors"
WTYPE = "q8_0"           # "q4_0" = smallest/lowest quality, "q8_0" = near-lossless, "default" = f16 (tight on 8 GB)
N_THREADS = 6
OUTPUT_DIR = "outputs"

DEFAULT_QUALITY = "score_9, score_8_up, score_7_up, masterpiece"
DEFAULT_PROMPT = "1girl, solo, long silver hair, blue eyes, sitting on bed, looking at viewer, detailed background, bedroom, soft lighting"
DEFAULT_NEGATIVE = "score_4, score_3, score_2, score_1, 3d, bad anatomy, bad hands, lowres, blurry, jpeg artifacts, watermark, text"

RESOLUTIONS = {
    "768 x 1024 (portrait)": (768, 1024),
    "832 x 1216 (tall portrait)": (832, 1216),
    "1024 x 1024 (square)": (1024, 1024),
    "1024 x 768 (landscape)": (1024, 768),
    "1216 x 832 (wide)": (1216, 832),
    "640 x 896 (low VRAM)": (640, 896),
}
# ------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
_lock = threading.Lock()  # one generation at a time on the GPU


def supported(func, **kwargs):
    """Keep only kwargs the installed version accepts."""
    params = inspect.signature(func).parameters
    dropped = [k for k in kwargs if k not in params]
    if dropped:
        print(f"[note] skipped unsupported args: {dropped}")
    return {k: v for k, v in kwargs.items() if k in params}


print("Loading model (this takes a while the first time)...")
sd = StableDiffusion(**supported(
    StableDiffusion.__init__,
    model_path=MODEL_PATH,
    vae_path=VAE_PATH,
    n_threads=N_THREADS,
    wtype=WTYPE,
    vae_decode_only=True,
    diffusion_flash_attn=True,
    verbose=False,
))
print("Model ready.")


def generate(prompt, negative, use_quality, resolution, steps, cfg, sampler,
             clip_skip, seed, batch, tiling):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    width, height = RESOLUTIONS[resolution]
    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    full_prompt = f"{DEFAULT_QUALITY}, {prompt}" if use_quality else prompt

    start = time.time()
    with _lock:
        images = sd.generate_image(**supported(
            StableDiffusion.generate_image,
            prompt=full_prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            sample_steps=int(steps),
            cfg_scale=float(cfg),
            sample_method=sampler,
            clip_skip=int(clip_skip),
            seed=seed,
            batch_count=int(batch),
            vae_tiling=bool(tiling),
        ))

    if not images:
        raise gr.Error("Generation failed (probably out of VRAM). "
                       "Lower the resolution or keep VAE tiling on.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, img in enumerate(images):
        img.save(os.path.join(OUTPUT_DIR, f"{stamp}_seed{seed + i}.png"))

    info = f"Done in {time.time() - start:.1f}s | seed {seed} | saved to ./{OUTPUT_DIR}"
    return images, info


def clear_all():
    return [], "", DEFAULT_PROMPT


with gr.Blocks(title="Local Pony Generator") as demo:
    gr.Markdown("## Local image generator (Vulkan / RX 6600)")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4)
            use_quality = gr.Checkbox(
                label=f"Prepend quality tags ({DEFAULT_QUALITY})", value=True)
            negative = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE, lines=3)
            resolution = gr.Dropdown(list(RESOLUTIONS), value="768 x 1024 (portrait)",
                                     label="Resolution")
            with gr.Row():
                steps = gr.Slider(10, 50, value=28, step=1, label="Steps")
                cfg = gr.Slider(1, 15, value=7.0, step=0.5, label="CFG scale")
            with gr.Row():
                sampler = gr.Dropdown(
                    ["euler_a", "euler", "dpm++2m", "dpm++2mv2", "heun", "lcm"],
                    value="euler_a", label="Sampler")
                clip_skip = gr.Slider(1, 2, value=2, step=1, label="Clip skip (Pony = 2)")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                batch = gr.Slider(1, 4, value=1, step=1, label="Batch count")
            tiling = gr.Checkbox(label="VAE tiling (prevents out-of-VRAM crash)", value=True)
            with gr.Row():
                go = gr.Button("Generate", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Result", columns=2, height=700, preview=True)
            status = gr.Textbox(label="Status", interactive=False)

    go.click(
        generate,
        inputs=[prompt, negative, use_quality, resolution, steps, cfg, sampler,
                clip_skip, seed, batch, tiling],
        outputs=[gallery, status],
    )
    clear.click(clear_all, outputs=[gallery, status, prompt])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", inbrowser=True)
