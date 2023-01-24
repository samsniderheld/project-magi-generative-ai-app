# Project Magi Scalable Generative AI Platform

This repo is a boilerplat for a container based scalable generative AI platform..

It's meant to demonstrate a basic docker based model as a service application.

Each service has it's own dockter container, along with a basic flask app that comunicates between the client, services, and storage buckets.

# Services

1. gradio_frontend: Communicates between services and uses gradio for UI
2. SD_txt2img_app: Uses huggingface diffusers to provide text2img MAAS.
3. SD_img2img_app: Uses huggingface diffusers to provide img2img MAAS.
4. SD_inpainting_app: Uses huggingface diffusers to provide inpainting MAAS

# installation
1. Go the the following link and launch a new VM using the Deep Learning VM

    (Deep Learning VM) [https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning]

    Use TensorFlow Enterprise 2.11 (CUDA 11.3) as the framework.

    Check Install NVIDIA GPU driver automatically on first startup.



2. enter the following command to download docker compose:

    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

3. The inpainting app needs a hugging face token. Generate one at (Hugging Face){https://huggingface.co/settings/tokens} and then in your VM enter:
    
    export HF_TOKEN={your token}

This will populate the code in the inpainting container.

4. git clone https://github.com/MightyHive/generative-labs-app.git

5. cd generative-labs-app

7. chmod +x run_docker.sh

8. sudo chmod 777 /usr/local/bin/docker-compose 

6. bash run_docker.sh






