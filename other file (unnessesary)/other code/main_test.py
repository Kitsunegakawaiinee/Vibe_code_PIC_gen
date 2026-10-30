import time
import inspect
from stable_diffusion_cpp import StableDiffusion

def supported(func, **kwargs):
    """Keep only kwargs this installed version accepts, and report the rest."""
    params = inspect.signature(func).parameters
    ok = {k: v for k, v in kwargs.items() if k in params}
    dropped = [k for k in kwargs if k not in params]
    if dropped:
        print(f"[note] not supported in your version, skipped: {dropped}")
    return ok

sd = StableDiffusion(**supported(
    StableDiffusion.__init__,
    model_path="model/ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
    n_threads=6,
    vae_decode_only=True,
    keep_vae_on_cpu=True,        # VAE decode in system RAM, avoids the 8 GB VRAM allocation
    diffusion_flash_attn=True,   # lowers attention memory/compute
    verbose=True,
))

quality_tags = ["score_9", "score_8_up", "score_7_up", "rating_explicit", "masterpiece"]
char_tags = ["1girl", "solo", "uncensored", "long silver hair", "blue eyes"]
pose_tags = ["naked", "sitting on bed", "looking at viewer"]
environment = ["detailed background", "bedroom", "soft lighting"]

prompt = ", ".join(quality_tags + char_tags + pose_tags + environment)
negative_prompt = "score_4, score_3, score_2, score_1, censored, 3d, bad anatomy, bad hands, lowres"

start_time = time.time()
images = sd.generate_image(**supported(
    StableDiffusion.generate_image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=1024,
    sample_steps=28,
    cfg_scale=7.0,
    sample_method="euler_a",
))
print(f"Done in {round(time.time() - start_time, 2)} seconds!")

if images:
    images[0].save("output/output_rx6600.png")
else:
    print("Generation failed, check the log above.")