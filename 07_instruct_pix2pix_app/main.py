"""main.py
The main flask interface for the inpainting SD app.
It takes the post variables and sends them to the
diffusers pipeline.
"""

import base64

from io import BytesIO
from PIL import Image
from flask import Flask, request
from model_interface import (get_instruct_pix2pix_pipeline, change_sampler, clean_from_gpu)

instruct_pipe = None

app = Flask(__name__)
@app.route("/generate", methods=['POST'])
def index():
    """The main entrypoint to this container"""
    global instruct_pipe

    prompt = request.form.get('prompt')
    sampling_steps = int(request.form.get('sampling_steps'))
    b64_img = request.form.get('init_img')
    cfg_scale = float(request.form.get('cfg_scale'))

    request_bytes_img = base64.b64decode(b64_img)
    reconstructed_bytes_img = BytesIO(request_bytes_img)
    input_image = Image.open(reconstructed_bytes_img)

    width, height = input_image.size
    input_image = input_image.resize((512,512))

    if instruct_pipe is None:
        print('loading pipeline...')
        instruct_pipe = get_instruct_pix2pix_pipeline()

    out_img = instruct_pipe(prompt,
        image=input_image,
        num_inference_steps = sampling_steps,
        guidance_scale = cfg_scale).images[0]

    out_img = out_img.resize((width,height))

    buff = BytesIO()
    out_img.save(buff, format="JPEG")
    response_img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return response_img_str

@app.route("/change_sampler", methods=['POST'])
def change_sampler_request():
    """The main entrypoint to this container"""

    new_sampler = request.form.get('new_sampler')
    global instruct_pipe
    instruct_pipe = change_sampler(instruct_pipe,new_sampler)
    return ('', 204)

@app.route("/clean_gpu", methods=['POST'])
def clean_gpu():
    """The main entrypoint to this container"""
    global instruct_pipe
    instruct_pipe = clean_from_gpu(instruct_pipe)
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
