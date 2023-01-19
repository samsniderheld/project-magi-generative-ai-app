"""main.py

This is the main entry pont for the gradio front end.
This file creates the frontend using a gradio interface.
Models interfaces are defined in the model_interface.py

"""
import base64
import gradio as gr
from io import BytesIO
from PIL import Image
from model_interface import send_img_request, change_sampler_request


samplers = [
    "euler_discreet",
    "DDIM",
    "LMS",
    "DPM",
    "PNDM",
    "DDPMS",
    "Euler_A"
]

sampler_urls = {

    "text2img": "http://sd_text2img_app:8000/change_sampler"
}

def change_sampler(sampler):
    """a function to call the change_sampler
    request in model_interface"""
    url = sampler_urls["text2img"]
    data = {"new_sampler": sampler}

    change_sampler_request(url,data)


def get_txt2img(prompt,negative_prompt,sampling_steps,width,height,cfg_scale):
    """
    This is the main interface between our base stable diffusion model and gradio.
    """

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


def get_inpaint(prompt,negative_prompt,input_image,mask,width,height,sampling_steps,cfg_scale):
    """
    This is the main interface between our inpainting model and gradio.
    """
    url = "http://sd_inpainting_app:10000/generate"

    raw_image = Image.fromarray(input_image).resize((512,512))
    img_buff = BytesIO()
    raw_image.save(img_buff, format="JPEG")
    b64_img = base64.b64encode(img_buff.getvalue()).decode("utf-8")


    raw_mask = Image.fromarray(mask).resize((512,512))
    mask_buff = BytesIO()
    raw_mask.save(mask_buff, format="JPEG")
    b64_mask = base64.b64encode(mask_buff.getvalue()).decode("utf-8")

    data = {"prompt":prompt,
        "negative_pompt":negative_prompt,
        "sampling_steps":sampling_steps,
        "init_img": b64_img,
        "init_mask": b64_mask,
        "width": width,
        "height": height,
        "cfg_scale":cfg_scale}

    image = send_img_request(url,data)

    return image


with gr.Blocks() as demo:

    with gr.Tab("text to img"):

        with gr.Row():

            with gr.Column():
                txt2img_sampler_input = gr.Dropdown(choices=samplers,label="Sampler",
                    value="euler_discreet")

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

                img2img_prompt_input = gr.Textbox(label="prompt")
                img2img_negative_prompt_input = gr.Textbox(label="negative prompt")
                img2img_input_img = gr.Image(label="input img")
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

                inpaint_prompt_input = gr.Textbox(label="prompt")
                inpaint_negative_prompt_input = gr.Textbox(label="negative prompt")
                inpaint_input_img = gr.Image(label="input img")
                inpaint_mask_img = gr.Image(label="mask img")
                inpaint_width_input = gr.Slider(512,1024,256,label="width")
                inpaint_height_input = gr.Slider(512,1024,256,label="height")
                inpaint_steps_input = gr.Slider(0, 150, value=50,
                    label="number of diffusion steps")
                inpaint_cfg_input = gr.Slider(0,30,value=7.5,label="cfg scale")

                inpaint_inputs = [
                    inpaint_prompt_input,
                    inpaint_negative_prompt_input,
                    inpaint_input_img,
                    inpaint_mask_img,
                    inpaint_width_input,
                    inpaint_height_input,
                    inpaint_steps_input,
                    inpaint_cfg_input,
                ]

            with gr.Column():

                inpaint_output = gr.Image()

        with gr.Row():

            inpaint_submit = gr.Button("Submit")

    txt2img_submit.click(get_txt2img,inputs=txt2img_inputs,outputs=txt2img_output)
    img2img_submit.click(get_img2img,inputs=img2img_inputs,outputs=img2img_output)
    inpaint_submit.click(get_inpaint,inputs=inpaint_inputs,outputs=inpaint_output)

    txt2img_sampler_input.change(fn=change_sampler, inputs=txt2img_sampler_input)

gradio_app = demo
