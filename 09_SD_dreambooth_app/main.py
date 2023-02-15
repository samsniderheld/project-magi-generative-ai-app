"""main.py
The main flask interface for the text2img SD app.
It takes the post variables and sends them to the
diffusers pipeline.
"""
import base64
import gc
import torch
import os
from io import BytesIO
from flask import Flask, request
from model_interface import (get_stable_diffusion_pipeline, change_sampler, clean_from_gpu)
from PIL import Image
import json
import subprocess
import time

txt2img_pipe = None
app = Flask(__name__)
@app.route("/train", methods=['POST'])
def index():
    """The main entrypoint to this container"""
    global txt2img_pipe
    start = time.time()

    model_name = request.form.get('model_name')
    instance_prompt = request.form.get('instance_prompt')
    class_prompt = request.form.get('class_prompt')
    steps = int(request.form.get('steps'))
    resolution = int(request.form.get('resolution'))

    concepts_list =[{
        "instance_prompt":instance_prompt,
        "class_prompt":class_prompt,
        "instance_data_dir":"tmp/data/zwx",
        "class_data_dir":"tmp/data/person"
        }]

    # instance_dir = os.path.join("tmp","data", concepts_list["instance_data_dir"])
    # class_dir = os.path.join("tmp","data", concepts_list["class_data_dir"])
    for concept in concepts_list:
        os.makedirs(concept["instance_data_dir"], exist_ok=True)  
        os.makedirs(concept["instance_data_dir"], exist_ok=True)  
    
    #images
    b64_images = request.form.getlist('b64_images')
    for i, b64_img in enumerate(b64_images):
        request_bytes = base64.b64decode(b64_img)
        reconstructed_bytes = BytesIO(request_bytes)
        input_image = Image.open(reconstructed_bytes)
        input_image.save(os.path.join(concepts_list[0]["instance_data_dir"], f'image_{i}.jpg'))    

    OUTPUT_DIR = "stable_diffusion_weights/zwx"

    with open("tmp/concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    script = f"""
        accelerate launch train_dreambooth.py \
            --pretrained_model_name_or_path={model_name} \
            --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
            --output_dir={OUTPUT_DIR} \
            --revision="fp16" \
            --with_prior_preservation --prior_loss_weight=1.0 \
            --seed=1337 \
            --resolution=512 \
            --train_batch_size=1 \
            --train_text_encoder \
            --mixed_precision="fp16" \
            --use_8bit_adam \
            --gradient_accumulation_steps=1 \
            --learning_rate=1e-6 \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --num_class_images=50 \
            --sample_batch_size=4 \
            --max_train_steps={steps} \
            --save_interval=10000 \
            --save_sample_prompt="photo of zwx person" \
            --concepts_list="tmp/concepts_list.json"
    """

    # !mkdir -p ~/.huggingface
    # HUGGINGFACE_TOKEN = "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" #@param {type:"string"}
    # !echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token

    print('running subprocess')
    process = subprocess.Popen(script, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, encoding='utf-8', errors='replace')

    while True:
        realtime_output = process.stdout.readline()

        if realtime_output == '' and process.poll() is not None:
            break

        if realtime_output:
            print(f"{realtime_output.strip()}")
            with open("logs.txt", 'a') as logs:
                logs.write(f"{realtime_output.strip()} \n")

    end = time.time()
    
    print("Total Process TIME", end-start)
    
    return 'Ok'

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
