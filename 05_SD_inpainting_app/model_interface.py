"""model_interface.py

Here we separate all the model interactions.
The purpose is to create a total separation between the frontend and models.
This is to keep everything modular so that models can be updated easily

"""
import os
import torch

from diffusers import StableDiffusionInpaintPipeline

from huggingface_hub import login

hf_token = os.environ['HF_TOKEN']

login(token=hf_token )

def get_stable_diffusion_inpainting_pipeline():
    """This is a wrapper to return
    a stable diffusion inpainting diffuser pipeline"""
    model_id = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    pipe = pipe.to("cuda")
    return pipe
