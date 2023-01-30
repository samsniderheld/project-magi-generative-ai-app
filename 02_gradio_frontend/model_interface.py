"""model_interface.py

Here we separate all the model interactions.
The purpose is to create a total separation between the frontend and models.
This is to keep everything modular so that models can be updated easily.

Since this is the front end app, only a api request is needed.

"""
import base64
import requests
import torch

from io import BytesIO
from PIL import Image


def send_img_request(url,data):
    """this is an api request to
    a stable diffusion containter"""
    #trigger detection app
    txt2img_url = url
    image = requests.post(txt2img_url, data=data)
    output = image.text
    response_bytes = base64.b64decode(output)
    reconstructed_bytes = BytesIO(response_bytes)
    response_image = Image.open(reconstructed_bytes)

    return response_image

def send_interrogator_request(url,data):
    """this is an api request to
    a stable diffusion containter"""
    #trigger detection app
    interrogator_url = url
    prompt = requests.post(interrogator_url, data=data)
    print(prompt)
    output = prompt.text

    return output

def change_sampler_request(change_sampler_url, data):
    """this is an api request to
    a change the sampler"""
    requests.post(change_sampler_url, data=data)
