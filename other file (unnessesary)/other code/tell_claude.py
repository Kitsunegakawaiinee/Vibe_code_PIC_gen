import inspect
from stable_diffusion_cpp import StableDiffusion
print(inspect.signature(StableDiffusion.__init__))
print(inspect.signature(StableDiffusion.generate_image))