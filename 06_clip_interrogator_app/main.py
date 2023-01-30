"""main.py
The main flask interface for the text2img SD app. 
It takes the post variables and sends them to the 
diffusers pipeline.
"""
""

import base64

from io import BytesIO
from PIL import Image
from flask import Flask, request, send_file
from model_interface import get_interrogator, inference, clean_from_gpu

ci = None

app = Flask(__name__)
@app.route("/interrogate", methods=['POST'])
def index():
    """The main entrypoint to this container"""
    global ci
    model_name = request.form.get('model_name')
    mode = request.form.get('mode')
    caption_max_lenght = int(float(request.form.get('caption_max_lenght')))
    caption_num_beams = int(float(request.form.get('caption_num_beams')))

    b64_img = request.form.get('image')

    request_bytes = base64.b64decode(b64_img)
    reconstructed_bytes = BytesIO(request_bytes)
    image = Image.open(reconstructed_bytes)

    if ci is None:
        print('getting interrogator...')
        ci = get_interrogator()    

    print(f'starting {model_name} clip interrogator...')
    prompt = inference(image, ci, mode, model_name, caption_max_lenght, caption_num_beams)

    return prompt

@app.route("/clean_gpu", methods=['POST'])
def clean_gpu():
    """The main entrypoint to this container"""
    global ci
    ci = clean_from_gpu(ci)
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
