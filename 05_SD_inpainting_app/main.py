"""main.py
The main flask interface for the inpainting SD app. 
It takes the post variables and sends them to the 
diffusers pipeline.
"""

import base64

from io import BytesIO
from PIL import Image
from flask import Flask, request, send_file
from model_interface import get_stable_diffusion_inpainting_pipeline

inpaint_pipe = get_stable_diffusion_inpainting_pipeline()

app = Flask(__name__)
@app.route("/generate", methods=['POST'])
def index():
    """The main entrypoint to this container"""

    prompt = request.form.get('prompt')
    negative_prompt = request.form.get('negative_prompt')
    sampling_steps = int(request.form.get('sampling_steps'))
    height = int(request.form.get('height'))
    width = int(request.form.get('width'))
    b64_img = request.form.get('init_img')
    b64_mask = request.form.get('init_mask')
    cfg_scale = float(request.form.get('cfg_scale'))

    request_bytes_img = base64.b64decode(b64_img)
    reconstructed_bytes_img = BytesIO(request_bytes_img)
    input_image = Image.open(reconstructed_bytes_img)

    request_bytes_mask = base64.b64decode(b64_mask)
    reconstructed_bytes_mask = BytesIO(request_bytes_mask)
    mask = Image.open(reconstructed_bytes_mask)

    out_img = inpaint_pipe(prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        mask_image=mask,
        width=width,
        height=height,
        num_inference_steps = sampling_steps,
        guidance_scale = cfg_scale).images[0]

    buff = BytesIO()
    out_img.save(buff, format="JPEG")
    response_img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return response_img_str

if __name__ == '__main__':
    app.run(debug=True)
