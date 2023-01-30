"""model_interface.py

Here we separate all the model interactions.
The purpose is to create a total separation between the frontend and models.
This is to keep everything modular so that models can be updated easily

"""
import torch
from clip_interrogator.clip_interrogator import Config, Interrogator
from PIL import Image
import gc

def get_interrogator():

    ci = Interrogator(Config(cache_path="cache", clip_model_path="cache"))
    return ci


def inference(image, ci, mode, clip_model_name, blip_max_length, blip_num_beams):
    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()
    ci.config.blip_max_length = int(blip_max_length)
    ci.config.blip_num_beams = int(blip_num_beams)

    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)

def clean_from_gpu(ci):
    print('removing model from memory...')
    del ci
    gc.collect()
    torch.cuda.empty_cache()

    return None