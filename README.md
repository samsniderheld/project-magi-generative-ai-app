# Project Magi Scalable Generative AI Platform

This repo is a boilerplat for a container based scalable generative AI platform..

It's meant to demonstrate a basic docker based model as a service application.

Each service has it's own dockter container, along with a basic flask app that comunicates between the client, services, and storage buckets.

# Services

1. gradio_frontend: Communicates between services and uses gradio for UI
2. SD_txt2img_app: Uses huggingface diffusers to provide text2img MAAS.
3. SD_img2img_app: Uses huggingface diffusers to provide img2img MAAS.
4. SD_inpainting_app: Uses huggingface diffusers to provide inpainting MAAS





