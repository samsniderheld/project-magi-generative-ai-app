"""model_interface.py

Here we separate all the model interactions.
The purpose is to create a total separation between the frontend and models.
This is to keep everything modular so that models can be updated easily

"""
import os
import torch
import gc

from diffusers import (StableDiffusionInstructPix2PixPipeline,
    EulerDiscreteScheduler, DDIMScheduler, LMSDiscreteScheduler, 
    DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler,
    EulerAncestralDiscreteScheduler)

from huggingface_hub import login

# hf_token = os.environ['HF_TOKEN']

# login(token=hf_token )



schedulers = {
    "euler_discreet": EulerDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "LMS": LMSDiscreteScheduler,
    "DPM": DPMSolverMultistepScheduler,
    "PNDM": PNDMScheduler,
    "DDPMS":  DDPMScheduler,
    "Euler_A": EulerDiscreteScheduler

}

def change_sampler(pipe, scheduler):
    """changes the scheduler"""
    new_scheduler = schedulers[scheduler]
    pipe.scheduler = new_scheduler.from_config(pipe.scheduler.config)
    return pipe

def get_instruct_pix2pix_pipeline():
    """This is a wrapper to return
    a stable diffusion instruct pix2pix pipeline"""
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda")

    return pipe

def clean_from_gpu(pipe):
    print('removing model from memory...')
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return None