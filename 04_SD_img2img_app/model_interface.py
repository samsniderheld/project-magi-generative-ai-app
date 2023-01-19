"""model_interface.py

Here we separate all the model interactions.
The purpose is to create a total separation between the frontend and models.
This is to keep everything modular so that models can be updated easily

"""
import os
import torch

from diffusers import (StableDiffusionPipeline, EulerDiscreteScheduler,
    StableDiffusionImg2ImgPipeline)

def get_stable_diffusion_img2img_pipeline():
    """This is a wrapper to return
    a stable diffusion img2img diffuser pipeline"""
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,
        revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe
