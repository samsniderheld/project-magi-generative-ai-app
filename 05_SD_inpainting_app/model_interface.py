"""model_interface.py

Here we separate all the model interactions.
The purpose is to create a total separation between the frontend and models.
This is to keep everything modular so that models can be updated easily

"""
import os
import torch
import gc 
from torchvision import transforms

from io import BytesIO
import requests
import PIL
from PIL import Image
import numpy as np
import uuid
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from clipseg.models.clipseg import CLIPDensePredT

from diffusers import (StableDiffusionInpaintPipeline,
    EulerDiscreteScheduler, DDIMScheduler, LMSDiscreteScheduler, 
    DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler,
    EulerAncestralDiscreteScheduler)

from huggingface_hub import login


hf_token = os.environ['HF_TOKEN']

login(token=hf_token )

transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      transforms.Resize((512, 512)),
])

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

def get_stable_diffusion_inpainting_pipeline():
    """This is a wrapper to return
    a stable diffusion inpainting diffuser pipeline"""
    model_id = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    return pipe

def get_mask_using_clip(img, word_mask):
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load('./clipseg/weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False)

    img = transform(img).unsqueeze(0)
    word_masks = [word_mask]
    print(word_mask)
    print(word_masks)
    with torch.no_grad():
        preds = model(img.repeat(len(word_masks),1,1,1), word_masks)[0]
    filename = f"{uuid.uuid4()}.png"
    plt.imsave(filename,torch.sigmoid(preds[0][0]))
    img2 = cv2.imread(filename)
    gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
    mask = Image.fromarray(np.uint8(bw_image)).convert('RGB')
    os.remove(filename)

    return mask 

def clean_from_gpu(pipe):
    print('removing model from memory...')
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return None