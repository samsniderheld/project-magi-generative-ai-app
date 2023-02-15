"""main.py
The main flask interface for the img2img SD app.
It takes the post variables and sends them to the
diffusers pipeline.
"""

import base64

from io import BytesIO
from PIL import Image
from flask import Flask, request
import torch
from model_interface import (get_stable_diffusion_img2img_pipeline, change_sampler, clean_from_gpu)

img2img_pipe = None

app = Flask(__name__)
@app.route("/generate", methods=['POST'])
def index():
    """The main entrypoint to this container"""
    global img2img_pipe

    prompt = request.form.get('prompt')
    negative_prompt = request.form.get('negative_prompt')
    sampling_steps = int(request.form.get('sampling_steps'))
    b64_img = request.form.get('init_img')

    request_bytes = base64.b64decode(b64_img)
    reconstructed_bytes = BytesIO(request_bytes)
    input_image = Image.open(reconstructed_bytes)

    cfg_scale = float(request.form.get('cfg_scale'))
    strength = float(request.form.get('strength'))

    if img2img_pipe is None:
        print('getting pipeline...')
        img2img_pipe = get_stable_diffusion_img2img_pipeline()
    
    with torch.inference_mode():
        image = img2img_pipe(prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            num_inference_steps = sampling_steps,
            guidance_scale = cfg_scale).images[0]

    buff = BytesIO()
    image.save(buff, format="JPEG")
    response_img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return response_img_str

@app.route("/change_sampler", methods=['POST'])
def change_sampler_request():
    """The main entrypoint to this container"""

    new_sampler = request.form.get('new_sampler')
    global img2img_pipe
    img2img_pipe = change_sampler(img2img_pipe,new_sampler)
    return ('', 204)

@app.route("/clean_gpu", methods=['POST'])
def clean_gpu():
    """The main entrypoint to this container"""
    global img2img_pipe
    img2img_pipe = clean_from_gpu(img2img_pipe)
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
