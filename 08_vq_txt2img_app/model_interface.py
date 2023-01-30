"""model_interface.py

Here we separate all the model interactions.
The purpose is to create a total separation between the frontend and models.
This is to keep everything modular so that models can be updated easily

"""
import torch
import gc

from diffusers.pipelines import VQDiffusionPipeline

def get_vq_diffusion_pipeline():
    """This is a wrapper to return
    a stable diffusion text2img diffuser pipeline"""
    model_id = "stabilityai/stable-diffusion-2"
    # Use the Euler scheduler here instead
    # scheduler = ?
    pipe = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq")
    pipe = pipe.to("cuda")
    return pipe

def clean_from_gpu(pipe):
    print('removing model from memory...')
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return None
