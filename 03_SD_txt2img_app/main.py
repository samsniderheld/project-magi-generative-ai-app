"""main.py
The main flask interface for the text2img SD app.
It takes the post variables and sends them to the
diffusers pipeline.
"""
import base64
import gc
import torch

from io import BytesIO
from flask import Flask, request
from model_interface import (get_stable_diffusion_pipeline, change_sampler, clean_from_gpu)

txt2img_pipe = None
app = Flask(__name__)
@app.route("/generate", methods=['POST'])
def index():
    """The main entrypoint to this container"""
    global txt2img_pipe
    prompt = request.form.get('prompt')
    negative_prompt = request.form.get('negative_prompt')
    sampling_steps = int(request.form.get('sampling_steps'))
    height = int(request.form.get('height'))
    width = int(request.form.get('width'))
    cfg_scale = float(request.form.get('cfg_scale'))

    if txt2img_pipe is None:
        print("getting txt2img pipeline...")
        txt2img_pipe = get_stable_diffusion_pipeline()

    with torch.inference_mode():
        image = txt2img_pipe(prompt,height,width,sampling_steps,
            cfg_scale,negative_prompt).images[0]
    
    buff = BytesIO()
    image.save(buff, format="JPEG")
    response_img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return response_img_str

@app.route("/change_sampler", methods=['POST'])
def change_sampler_request():
    """The main entrypoint to this container"""

    new_sampler = request.form.get('new_sampler')
    global txt2img_pipe
    txt2img_pipe = change_sampler(txt2img_pipe,new_sampler)
    return ('', 204)

@app.route("/clean_gpu", methods=['POST'])
def clean_gpu():
    """The main entrypoint to this container"""
    global txt2img_pipe
    txt2img_pipe = clean_from_gpu(txt2img_pipe)
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
