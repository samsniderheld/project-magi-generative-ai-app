"""This is a prototype test functionality.
More thinking about how to test theses apps is
needed"""

from model_interface import get_stable_diffusion_pipeline

txt2img_pipe = get_stable_diffusion_pipeline()

prompt = "a puppy"
negative_prompt = "a cat"
sampling_steps = "50"
height = "512"
width = "512"
cfg_scale = "7.5"

image = txt2img_pipe(prompt,height,width,sampling_steps,
    cfg_scale,negative_prompt).images[0]

print(type(image))
print(image.size)