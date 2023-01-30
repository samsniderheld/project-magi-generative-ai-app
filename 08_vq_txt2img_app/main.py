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
from model_interface import (get_vq_diffusion_pipeline, clean_from_gpu)

txt2img_vq_pipe = None
app = Flask(__name__)
@app.route("/generate", methods=['POST'])
def index():
    """The main entrypoint to this container"""
    global txt2img_vq_pipe
    prompt = request.form.get('prompt')
    sampling_steps = int(request.form.get('sampling_steps'))
    cfg_scale = float(request.form.get('cfg_scale'))

    if txt2img_vq_pipe is None:
        print("getting vq_txt2img pipeline...")
        txt2img_vq_pipe = get_vq_diffusion_pipeline()

    image = txt2img_vq_pipe(prompt=prompt,num_inference_steps=sampling_steps,
        guidance_scale=cfg_scale).images[0]
    
    buff = BytesIO()
    image.save(buff, format="JPEG")
    response_img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return response_img_str

@app.route("/clean_gpu", methods=['POST'])
def clean_gpu():
    """The main entrypoint to this container"""
    global txt2img_vq_pipe
    txt2img_vq_pipe = clean_from_gpu(txt2img_vq_pipe)
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
