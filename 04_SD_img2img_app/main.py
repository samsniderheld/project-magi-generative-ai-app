"""main.py
The main flask interface for the img2img SD app. 
It takes the post variables and sends them to the 
diffusers pipeline.
"""

import base64

from io import BytesIO
from PIL import Image
from flask import Flask, request, send_file
from model_interface import get_stable_diffusion_img2img_pipeline

img2img_pipe = get_stable_diffusion_img2img_pipeline()

app = Flask(__name__)
@app.route("/generate", methods=['POST'])
def index():
    """The main entrypoint to this container"""

    prompt = request.form.get('prompt')
    negative_prompt = request.form.get('negative_prompt')
    sampling_steps = int(request.form.get('sampling_steps'))
    b64_img = request.form.get('init_img')

    request_bytes = base64.b64decode(b64_img)
    reconstructed_bytes = BytesIO(request_bytes)
    input_image = Image.open(reconstructed_bytes)

    cfg_scale = float(request.form.get('cfg_scale'))
    strength = float(request.form.get('strength'))

    image = img2img_pipe(prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=input_image,
        strength=strength,
        num_inference_steps = sampling_steps,
        guidance_scale = cfg_scale).images[0]

    buff = BytesIO()
    image.save(buff, format="JPEG")
    response_img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return response_img_str

if __name__ == '__main__':
    app.run(debug=True)
