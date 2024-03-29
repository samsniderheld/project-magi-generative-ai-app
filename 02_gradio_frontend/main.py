"""main.py

This is the main entry pont for the gradio front end.
This file creates the frontend using a gradio interface.
Models interfaces are defined in the model_interface.py
.
"""
import base64
import gradio as gr
from io import BytesIO
from PIL import Image
import gc
import requests
import numpy as np

from model_interface import (send_img_request,
  change_sampler_request, send_interrogator_request)


samplers = [
    "Euler_A",
    "euler_discreet",
    "DDIM",
    "LMS",
    "DPM",
    "PNDM",
    "DDPMS"    
]

sampler_urls = {

    "text2img": "http://sd_text2img_app:8000/change_sampler",
    "img2img": "http://sd_img2img_app:9000/change_sampler",
    "inpainting": "http://sd_inpainting_app:10000/change_sampler",

}

models_db = [
    "runwayml/stable-diffusion-v1-5",
]

"""hacky fix until I solve
AttributeError: 'str' object has no attribute '_id'
"""

def change_sampler_txt2img(sampler):
    """a function to call the change_sampler
    request in model_interface"""
    url = sampler_urls["text2img"]
    data = {"new_sampler": sampler}

    change_sampler_request(url,data)


def change_sampler_img2img(sampler):
    """a function to call the change_sampler
    request in model_interface"""
    url = sampler_urls["img2img"]
    data = {"new_sampler": sampler}

    change_sampler_request(url,data)

def change_sampler_inpainting(sampler):
    """a function to call the change_sampler
    request in model_interface"""
    url = sampler_urls["inpainting"]
    data = {"new_sampler": sampler}

    change_sampler_request(url,data)


clean_urls = {

    "txt2img": "http://sd_text2img_app:8000/clean_gpu",
    "img2img": "http://sd_img2img_app:9000/clean_gpu",
    "inpainting": "http://sd_inpainting_app:10000/clean_gpu",
    "interrogator":  "http://clip_interrogator_app:11000/clean_gpu",
    "instruct": "http://instruct_pix2pix_app:12000/clean_gpu",

}

def clean_gpu(service):
    """a function to clean the GPU after interrogator"""
    # [requests.post(url) for url in clean_urls if url != service]
    resultset = [url for key, url in clean_urls.items() if key not in service]

    for key, url_value in clean_urls.items():
        if key not in service:
            print(url_value)
            requests.post(url_value)

def get_txt2img(prompt,negative_prompt,sampling_steps,width,height,cfg_scale):
    """
    This is the main interface between our base stable diffusion model and gradio.
    """
    clean_gpu('txt2img')
    url = "http://sd_text2img_app:8000/generate"

    data = {"prompt":prompt,
        "negative_pompt":negative_prompt,
        "sampling_steps":sampling_steps,
        "width":width,
        "height":height,
        "cfg_scale":cfg_scale}

    image = send_img_request(url,data)

    return image


def get_img2img(prompt,negative_prompt,input_image,strength,sampling_steps,cfg_scale):
    """
    This is the main interface between our img2img model and gradio.
    """
    clean_gpu("img2img")
    url = "http://sd_img2img_app:9000/generate"

    raw_image = Image.fromarray(input_image)

    buff = BytesIO()
    raw_image.save(buff, format="JPEG")
    b64_img = base64.b64encode(buff.getvalue()).decode("utf-8")

    data = {"prompt":prompt,
        "negative_pompt":negative_prompt,
        "sampling_steps":sampling_steps,
        "init_img": b64_img,
        "strength": strength,
        "cfg_scale":cfg_scale}

    image = send_img_request(url,data)

    return image

def swap_word_mask(radio_option):
    if(radio_option == "type what to mask below"):
        return gr.update(interactive=True, placeholder="A cat")
    else:
        return gr.update(interactive=False, placeholder="Disabled")

def get_inpaint(prompt,negative_prompt,input_image_dict,sampling_steps,cfg_scale, inpaint_input_radio, inpaint_input_word_mask):
    """
    This is the main interface between our inpainting model and gradio.
    """
    clean_gpu("inpainting")
    url = "http://sd_inpainting_app:10000/generate"
    
    #raw_image = Image.fromarray(input_image_dict["image"]).convert("RGB").resize((512,512)) #
    raw_image = input_image_dict["image"].convert("RGB").resize((512, 512))
    img_buff = BytesIO()
    raw_image.save(img_buff, format="JPEG")
    b64_img = base64.b64encode(img_buff.getvalue()).decode("utf-8")


    #raw_mask = Image.fromarray(input_image_dict["mask"]).convert("RGB").resize((512,512))
    raw_mask = input_image_dict["mask"].convert("RGB").resize((512, 512))

    mask_buff = BytesIO()
    raw_mask.save(mask_buff, format="JPEG")
    b64_mask = base64.b64encode(mask_buff.getvalue()).decode("utf-8")

    data = {"prompt":prompt,
        "negative_pompt":negative_prompt,
        "sampling_steps":sampling_steps,
        "init_img": b64_img,
        "init_mask": b64_mask,
        #"width": width,
        #"height": height,
        "cfg_scale":cfg_scale,
        "radio": inpaint_input_radio, 
        "word_mask": inpaint_input_word_mask}

    image = send_img_request(url,data)

    return image


def get_instruct_pix2pix(prompt,input_image,sampling_steps,cfg_scale):
    """
    This is the main interface between our inpainting model and gradio.
    """
    clean_gpu("instruct")
    url = "http://instruct_pix2pix_app:12000/generate"

    raw_image = Image.fromarray(input_image)
    img_buff = BytesIO()
    raw_image.save(img_buff, format="JPEG")
    b64_img = base64.b64encode(img_buff.getvalue()).decode("utf-8")

    data = {"prompt":prompt,
        "sampling_steps":sampling_steps,
        "init_img": b64_img,
        "cfg_scale":cfg_scale}

    image = send_img_request(url,data)

    return image
                
def get_clip_interrogator(image, model_name, mode, caption_max_lenght, caption_num_beams):
    """
    This is the main interface between our interogator model and gradio.
    """
    clean_gpu("interrogator")

    url = "http://clip_interrogator_app:11000/interrogate"

    image = Image.fromarray(image)

    buff = BytesIO()
    image.save(buff, format="JPEG")
    b64_image = base64.b64encode(buff.getvalue()).decode("utf-8")

    data = {"image": b64_image,
        "model_name":model_name,
        "mode":mode,
        "caption_max_lenght": caption_max_lenght,
        "caption_num_beams": caption_num_beams
        }

    prompt = send_interrogator_request(url,data)
  
    return prompt

def get_train_db(model_name, instance_prompt, class_prompt, steps, resolution, images):
    """
    This is the main interface between our interogator model and gradio.
    """
    #clean_gpu("dreambooth")

    url = "http://SD_dreambooth_app:13000/train"

    b64_images = []
    for i, image in enumerate(images):
        print(f'processing image {i}')
        image = Image.open(image)

        buff = BytesIO()
        image.save(buff, format="JPEG")
        b64_image = base64.b64encode(buff.getvalue()).decode("utf-8")
        b64_images.append(b64_image)

    data = {"model_name": model_name,
        "instance_prompt":instance_prompt,
        "class_prompt":class_prompt,
        "steps": steps,
        "resolution": resolution,
        "b64_images": b64_images
        }

    msg  = send_interrogator_request(url,data)

    return msg

def upload_preview_gallery(file_imgs_input):
    imgs = []
    for img_input in file_imgs_input:
        imgs.append(Image.open(img_input))
    
    return imgs


# def get_vq_txt2img(prompt,sampling_steps,cfg_scale):
#     """
#     This is the main interface between our base stable diffusion model and gradio.
#     """
#     #clean_gpu('vq_txt2img')
#     url = "http://vq_txt2img_app:13000/generate"

#     data = {"prompt":prompt,
#         "sampling_steps":sampling_steps,
#         "cfg_scale":cfg_scale}

#     image = send_img_request(url,data)

#     return image

image_blocks = gr.Blocks()
with gr.Blocks() as demo:

    with gr.Tab("text to img"):

        with gr.Row():

            with gr.Column():
                txt2img_sampler_dropdown = gr.Dropdown(choices=samplers,label="Sampler",
                    value="Euler_A")

                txt2img_prompt_input = gr.Textbox(label="prompt")
                txt2img_negative_prompt_input = gr.Textbox(label="negative prompt")
                txt2img_steps_input = gr.Slider(0, 150, value=50,
                    label="number of diffusion steps")
                txt2img_width_input = gr.Slider(512,1024,256,label="width")
                txt2img_height_input = gr.Slider(512,1024,256,label="height")
                txt2img_cfg_input = gr.Slider(0,30,value=7.5,label="cfg scale")

                txt2img_inputs = [
                    txt2img_prompt_input,
                    txt2img_negative_prompt_input,
                    txt2img_steps_input,
                    txt2img_width_input,
                    txt2img_height_input,
                    txt2img_cfg_input,
                ]

            with gr.Column():

                txt2img_output = gr.Image()

        with gr.Row():

            txt2img_submit = gr.Button("Submit")

    with gr.Tab("img to img"):

        with gr.Row():

            with gr.Column():

                img2img_sampler_dropdown = gr.Dropdown(choices=samplers,label="Sampler",
                    value="euler_discreet")

                img2img_prompt_input = gr.Textbox(label="prompt")
                img2img_negative_prompt_input = gr.Textbox(label="negative prompt")
                #img2img_input_img = gr.Image(label="input img")
                img2img_input_img = gr.Image(tool='color-sketch', type='numpy', label='Input image', source='upload', interactive=True)
                img2img_strength_input = gr.Slider(0, 1, value=.5,
                    label="strength")
                img2img_steps_input = gr.Slider(0, 150, value=50,
                    label="number of diffusion steps")
                img2img_cfg_input = gr.Slider(0,30,value=7.5,label="cfg scale")

                img2img_inputs = [
                    img2img_prompt_input,
                    img2img_negative_prompt_input,
                    img2img_input_img,
                    img2img_strength_input,
                    img2img_steps_input,
                    img2img_cfg_input,
                ]

            with gr.Column():

                img2img_output = gr.Image()

        with gr.Row():

            img2img_submit = gr.Button("Submit")

    with gr.Tab("inpaint"):

        with gr.Row():

            with gr.Column():

                inpainting_sampler_drowdown = gr.Dropdown(choices=samplers,label="Sampler",
                    value="euler_discreet")

                inpaint_prompt_input = gr.Textbox(label="prompt")
                inpaint_negative_prompt_input = gr.Textbox(label="negative prompt")
                #inpaint_input_img = gr.Image(label="input img")
                #inpaint_mask_img = gr.Image(label="mask img")
                inpaint_input_img = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload")#.style(height=400)
                #inpaint_width_input = gr.Slider(512,1024,256,label="width")
                #inpaint_height_input = gr.Slider(512,1024,256,label="height")
                with gr.Box(elem_id="mask_radio").style(border=False):
                    inpaint_input_radio = gr.Radio(["draw a mask above", "type what to mask below"], value="draw a mask above", show_label=False, interactive=True).style(container=False)
                    inpaint_input_word_mask = gr.Textbox(label = "What to find in your image", interactive=False, elem_id="word_mask", placeholder="Disabled").style(container=False)
                inpaint_steps_input = gr.Slider(0, 150, value=50, label="number of diffusion steps")
                inpaint_cfg_input = gr.Slider(0,30,value=7.5,label="cfg scale")
                
                inpaint_input_radio.change(fn=swap_word_mask, inputs=inpaint_input_radio, outputs=inpaint_input_word_mask,show_progress=False)

                inpaint_inputs = [
                    inpaint_prompt_input,
                    inpaint_negative_prompt_input,
                    inpaint_input_img,
                    #inpaint_mask_img,
                    #inpaint_width_input,
                    #inpaint_height_input,
                    inpaint_steps_input,
                    inpaint_cfg_input,
                    inpaint_input_radio,
                    inpaint_input_word_mask,
                ]

            with gr.Column():

                inpaint_output = gr.Image()

        with gr.Row():

            inpaint_submit = gr.Button("Submit")

    with gr.Tab("instruct"):

        with gr.Row():

            with gr.Column():

                instruct_sampler_drowdown = gr.Dropdown(choices=samplers,label="Sampler",
                    value="euler_discreet")

                instruct_prompt_input = gr.Textbox(label="prompt")
                instruct_input_img = gr.Image(label="input img")
                instruct_steps_input = gr.Slider(0, 150, value=50,
                    label="number of diffusion steps")
                instruct_cfg_input = gr.Slider(0,30,value=7.5,label="cfg scale")

                instruct_inputs = [
                    instruct_prompt_input,
                    instruct_input_img,
                    instruct_steps_input,
                    instruct_cfg_input,
                ]

            with gr.Column():

                instruct_output = gr.Image()

        with gr.Row():

            instruct_submit = gr.Button("Submit")

    with gr.Tab("interrogator"):

        with gr.Row():

            with gr.Column():
                interrogator_img_input = gr.Image(label="input img")
                interrogator_model_name_input = gr.Dropdown(['ViT-L-14/openai', 'ViT-H-14/laion2b_s32b_b79k'], value='ViT-L-14/openai', label='CLIP Model')
                #interrogator_model_name_input = gr.Radio(['ViT-L-14/openai', 'ViT-H-14/laion2b_s32b_b79k'], label='', value='ViT-L-14/openai')  
                interrogator_mode_input = gr.Radio(['best', 'fast'], label='', value='best')
                #interrogator_best_max_flavors_input = gr.Number(value=16, label='best mode max flavors')
                interrogator_caption_max_length_input = gr.Number(value=32, label='Caption Max Length')
                interrogator_caption_num_beams_input = gr.Number(value=64, label='Caption Num Beams')

                interrogator_inputs = [
                    interrogator_img_input,
                    interrogator_model_name_input,
                    interrogator_mode_input,
                    #interrogator_best_max_flavors_input,
                    interrogator_caption_max_length_input,
                    interrogator_caption_num_beams_input
                ]

            with gr.Column():

                interrogator_output = gr.outputs.Textbox(label="Output")

        with gr.Row():

            interrogator_submit = gr.Button("Submit")

    with gr.Tab("dreambooth"):

        with gr.Row():

            with gr.Column():
                db_models_db_dropdown = gr.Dropdown(choices=models_db,label="Models",
                    value="runwayml/stable-diffusion-v1-5")

                db_instance_prompt_input = gr.Textbox(label="instance prompt", value="photo of zwx person")
                db_class_prompt_input = gr.Textbox(label="class prompt", value="photo of a person")
                db_steps_input = gr.Slider(300, 2500, 50, label="max training steps steps") 
                db_resolution_input = gr.Slider(512,1024,256,label="resolution")

            with gr.Column():

                db_images_input = gr.File(file_count="multiple")
                db_gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[7], height='auto')
                db_upload = gr.Button("Upload preview")
                

        with gr.Row():

            db_inputs = [
                db_models_db_dropdown,
                db_instance_prompt_input,
                db_class_prompt_input,
                db_steps_input,
                db_resolution_input,
                db_images_input,
            ]

            db_submit = gr.Button("Train")
            db_status_output = gr.Textbox(value="")

    # with gr.Tab("VQ text to img"):

    #     with gr.Row():

    #         with gr.Column():
    #             txt2img_vq_prompt_input = gr.Textbox(label="prompt")
    #             txt2img_vq_steps_input = gr.Slider(0, 150, value=100,
    #                 label="number of diffusion steps")
    #             txt2img_vq_cfg_input = gr.Slider(0,30,value=7.5,label="cfg scale")

    #             txt2img_vq_inputs = [
    #                 txt2img_vq_prompt_input,
    #                 txt2img_vq_steps_input,
    #                 txt2img_vq_cfg_input,
    #             ]

    #         with gr.Column():

    #             txt2img_vq_output = gr.Image()

    #     with gr.Row():

    #         txt2img_vq_submit = gr.Button("Submit")
    
    # end tabs
    txt2img_submit.click(get_txt2img,inputs=txt2img_inputs,outputs=txt2img_output)
    img2img_submit.click(get_img2img,inputs=img2img_inputs,outputs=img2img_output)
    #inpaint_submit.click(get_inpaint,inputs=inpaint_inputs,outputs=inpaint_output)
    inpaint_submit.click(fn=get_inpaint, inputs=inpaint_inputs, outputs=inpaint_output)
    instruct_submit.click(get_instruct_pix2pix,inputs=instruct_inputs,outputs=instruct_output)
    interrogator_submit.click(get_clip_interrogator, inputs=interrogator_inputs,outputs=interrogator_output)
    db_upload.click(upload_preview_gallery, inputs=db_images_input, outputs=db_gallery)
    db_submit.click(get_train_db, inputs=db_inputs, outputs=db_status_output)
    # txt2img_vq_submit.click(get_vq_txt2img,inputs=txt2img_vq_inputs,outputs=txt2img_vq_output)

    txt2img_sampler_dropdown.change(fn=change_sampler_txt2img, inputs=txt2img_sampler_dropdown)
    img2img_sampler_dropdown.change(fn=change_sampler_img2img, inputs=img2img_sampler_dropdown )
    inpainting_sampler_drowdown.change(fn=change_sampler_inpainting, inputs=inpainting_sampler_drowdown)


gradio_app = demo
