import os
import gc
import time
import random
import inspect
import threading
from datetime import datetime

import gradio as gr
from PIL import Image, ImageFilter
from stable_diffusion_cpp import StableDiffusion

# ----------------------------- CONFIG -----------------------------
MODEL_DIR = "model"
OUTPUT_DIR = "outputs"
N_THREADS = 6
DEFAULT_WTYPE = "q8_0"   # "q4_0" smallest, "q5_0", "q8_0" near-lossless, "default" = f16
MODEL_EXTS = (".safetensors", ".gguf", ".ckpt")

# The "Detail upscale" method (img2img refine) needs the VAE encoder, which uses
# extra VRAM. Leave False on 8 GB cards unless you want to try it.
ENABLE_DETAIL_UPSCALE = False

RESOLUTIONS = {
    "768 x 1024 (portrait)": (768, 1024),
    "832 x 1216 (tall portrait)": (832, 1216),
    "1024 x 1024 (square)": (1024, 1024),
    "1024 x 768 (landscape)": (1024, 768),
    "1216 x 832 (wide)": (1216, 832),
    "640 x 896 (low VRAM)": (640, 896),
}

# Auto presets, picked from the model filename ("pony" in name -> pony, else illustrious)
PRESETS = {
    "pony": dict(
        prefix="score_9, score_8_up, score_7_up, masterpiece",
        suffix="",
        negative="score_4, score_3, score_2, score_1, 3d, bad anatomy, bad hands, lowres, blurry, jpeg artifacts, watermark, text",
        cfg=6.5, sampler="euler_a", scheduler="default",
    ),
    "illustrious": dict(
        prefix="",
        suffix="masterpiece, best quality, amazing quality",
        negative="bad quality, worst quality, worst detail, sketch, censor",
        cfg=5.5, sampler="dpm++2m", scheduler="karras",
    ),
}
DEFAULT_PROMPT = "1girl, solo, adult, long silver hair, blue eyes, soft smile, looking at viewer, sitting on bed, bedroom, warm lighting"
# ------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
_lock = threading.Lock()  # one GPU job at a time
state = {"sd": None, "path": None, "wtype": None, "kind": "illustrious"}


def supported(func, **kwargs):
    """Keep only kwargs the installed version accepts."""
    params = inspect.signature(func).parameters
    dropped = [k for k in kwargs if k not in params]
    if dropped:
        print(f"[note] skipped unsupported args: {dropped}")
    return {k: v for k, v in kwargs.items() if k in params}


def list_models():
    files = []
    for f in sorted(os.listdir(MODEL_DIR)):
        if f.lower().endswith(MODEL_EXTS) and "vae" not in f.lower():
            files.append(f)
    return files


def kind_of(name):
    return "pony" if "pony" in name.lower() else "illustrious"


def load_model(name, wtype):
    """Load (or reload) a model. Frees the previous one first."""
    if not name:
        raise gr.Error(f"No model found. Put a .safetensors file in ./{MODEL_DIR}")
    path = os.path.join(MODEL_DIR, name)
    with _lock:
        if state["sd"] is not None and state["path"] == path and state["wtype"] == wtype:
            return
        if state["sd"] is not None:
            try:
                state["sd"].close()
            except Exception:
                pass
            state["sd"] = None
            gc.collect()
        print(f"Loading {name} ({wtype}) ...")
        state["sd"] = StableDiffusion(**supported(
            StableDiffusion.__init__,
            model_path=path,
            n_threads=N_THREADS,
            wtype=wtype,
            vae_decode_only=not ENABLE_DETAIL_UPSCALE,
            diffusion_flash_attn=True,
            verbose=False,
        ))
        state["path"], state["wtype"], state["kind"] = path, wtype, kind_of(name)
        print("Model ready.")


def on_model_change(name, wtype):
    load_model(name, wtype)
    p = PRESETS[state["kind"]]
    msg = f"Loaded: {name} ({wtype}) | preset: {state['kind']}"
    return msg, p["negative"], p["cfg"], p["sampler"], p["scheduler"]


def refresh_models():
    return gr.update(choices=list_models())


def build_prompt(prompt, auto_tags):
    if not auto_tags:
        return prompt
    p = PRESETS[state["kind"]]
    return ", ".join(x for x in (p["prefix"], prompt, p["suffix"]) if x)


def generate(prompt, negative, auto_tags, resolution, steps, cfg, sampler, scheduler,
             clip_skip, seed, batch, tiling):
    if state["sd"] is None:
        raise gr.Error("No model loaded.")
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    width, height = RESOLUTIONS[resolution]
    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    start = time.time()
    with _lock:
        images = state["sd"].generate_image(**supported(
            StableDiffusion.generate_image,
            prompt=build_prompt(prompt, auto_tags),
            negative_prompt=negative,
            width=width,
            height=height,
            sample_steps=int(steps),
            cfg_scale=float(cfg),
            sample_method=sampler,
            scheduler=scheduler,
            clip_skip=int(clip_skip),
            seed=seed,
            batch_count=int(batch),
            vae_tiling=bool(tiling),
        ))

    if not images:
        raise gr.Error("Generation failed (probably out of VRAM). "
                       "Lower the resolution or keep VAE tiling on.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for i, img in enumerate(images):
        path = os.path.join(OUTPUT_DIR, f"{stamp}_seed{seed + i}.png")
        img.save(path)
        paths.append(path)

    info = f"Done in {time.time() - start:.1f}s | seed {seed} | saved to ./{OUTPUT_DIR}"
    return paths, info, paths, 0


def on_select(evt: gr.SelectData):
    return evt.index


def upscale(paths, idx, method, factor, denoise, prompt, negative, auto_tags,
            steps, cfg, sampler, scheduler, clip_skip, tiling):
    if not paths:
        raise gr.Error("Generate an image first, then click the one you want to upscale.")
    idx = idx if isinstance(idx, int) and 0 <= idx < len(paths) else 0
    src = paths[idx]
    img = Image.open(src).convert("RGB")
    factor = float(factor)
    nw, nh = int(img.width * factor), int(img.height * factor)

    start = time.time()
    if method.startswith("Fast"):
        out = img.resize((nw, nh), Image.LANCZOS)
        out = out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=70, threshold=2))
    else:
        if not ENABLE_DETAIL_UPSCALE or state["sd"] is None:
            raise gr.Error("Detail upscale is off. Set ENABLE_DETAIL_UPSCALE = True at the top "
                           "of main_ui.py and restart (uses more VRAM).")
        nw, nh = (nw // 64) * 64, (nh // 64) * 64  # SDXL likes multiples of 64
        base = img.resize((nw, nh), Image.LANCZOS)
        with _lock:
            res = state["sd"].generate_image(**supported(
                StableDiffusion.generate_image,
                prompt=build_prompt(prompt, auto_tags),
                negative_prompt=negative,
                init_image=base,
                width=nw,
                height=nh,
                strength=float(denoise),
                sample_steps=int(steps),
                cfg_scale=float(cfg),
                sample_method=sampler,
                scheduler=scheduler,
                clip_skip=int(clip_skip),
                seed=random.randint(0, 2**31 - 1),
                vae_tiling=True,
            ))
        if not res:
            raise gr.Error("Detail upscale failed (probably out of VRAM). Try a smaller factor.")
        out = res[0]

    stem = os.path.splitext(os.path.basename(src))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_upscale_x{factor:g}.png")
    out.save(out_path)
    return out_path, f"Upscaled to {out.width}x{out.height} in {time.time() - start:.1f}s | saved: {out_path}"


def clear_all():
    return [], "", DEFAULT_PROMPT, [], 0, None


# ----------------------------- STARTUP -----------------------------
models = list_models()
if not models:
    raise SystemExit(f"No models found in ./{MODEL_DIR}. Put a .safetensors checkpoint there and rerun.")
load_model(models[0], DEFAULT_WTYPE)
P0 = PRESETS[state["kind"]]

with gr.Blocks(title="Local Anime Generator") as demo:
    gr.Markdown("## Local image generator (Vulkan)")
    paths_state = gr.State([])
    sel_state = gr.State(0)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                model_dd = gr.Dropdown(models, value=models[0], label=f"Model (from ./{MODEL_DIR})", scale=4)
                wtype_dd = gr.Dropdown(["q4_0", "q5_0", "q8_0", "default"], value=DEFAULT_WTYPE,
                                       label="Precision", scale=2)
                refresh = gr.Button("Refresh", scale=1)
            prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4)
            auto_tags = gr.Checkbox(label="Add quality tags automatically (matches the selected model)", value=True)
            negative = gr.Textbox(label="Negative prompt", value=P0["negative"], lines=3)
            resolution = gr.Dropdown(list(RESOLUTIONS), value="768 x 1024 (portrait)", label="Resolution")
            with gr.Row():
                steps = gr.Slider(10, 50, value=28, step=1, label="Steps")
                cfg = gr.Slider(1, 15, value=P0["cfg"], step=0.5, label="CFG scale")
            with gr.Row():
                sampler = gr.Dropdown(
                    ["euler_a", "euler", "dpm++2m", "dpm++2mv2", "heun", "lcm"],
                    value=P0["sampler"], label="Sampler")
                scheduler = gr.Dropdown(["default", "karras", "discrete", "exponential"],
                                        value=P0["scheduler"], label="Scheduler")
                clip_skip = gr.Slider(1, 2, value=2, step=1, label="Clip skip")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                batch = gr.Slider(1, 4, value=1, step=1, label="Batch count")
            tiling = gr.Checkbox(label="VAE tiling (prevents out-of-VRAM crash)", value=True)
            with gr.Row():
                go = gr.Button("Generate", variant="primary")
                clear = gr.Button("Clear")
            status = gr.Textbox(label="Status", interactive=False,
                                value=f"Loaded: {models[0]} ({DEFAULT_WTYPE}) | preset: {state['kind']}")

        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Result (click an image to select it)", columns=2,
                                 height=600, preview=False)
            gr.Markdown("### Upscale selected image")
            with gr.Row():
                up_method = gr.Dropdown(
                    ["Fast (Lanczos + sharpen)", "Detail (img2img refine)"],
                    value="Fast (Lanczos + sharpen)", label="Method")
                up_factor = gr.Dropdown(["1.5", "2", "3", "4"], value="2", label="Factor")
                up_denoise = gr.Slider(0.2, 0.6, value=0.35, step=0.05, label="Detail strength")
            up_btn = gr.Button("Upscale", variant="secondary")
            up_img = gr.Image(label="Upscaled result", type="filepath")

    shared_model_outputs = [status, negative, cfg, sampler, scheduler]
    model_dd.change(on_model_change, [model_dd, wtype_dd], shared_model_outputs)
    wtype_dd.change(on_model_change, [model_dd, wtype_dd], shared_model_outputs)
    refresh.click(refresh_models, outputs=model_dd)

    go.click(
        generate,
        inputs=[prompt, negative, auto_tags, resolution, steps, cfg, sampler, scheduler,
                clip_skip, seed, batch, tiling],
        outputs=[gallery, status, paths_state, sel_state],
    )
    gallery.select(on_select, outputs=sel_state)
    up_btn.click(
        upscale,
        inputs=[paths_state, sel_state, up_method, up_factor, up_denoise, prompt, negative,
                auto_tags, steps, cfg, sampler, scheduler, clip_skip, tiling],
        outputs=[up_img, status],
    )
    clear.click(clear_all, outputs=[gallery, status, prompt, paths_state, sel_state, up_img])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", inbrowser=True)
