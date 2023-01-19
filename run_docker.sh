#!/bin/bash
echo "killing old docker processes"
docker kill $(docker ps -q)
docker rm $(docker ps -a -q)

echo "building and running docker containers"
docker-compose up --build -d

# echo "running tests"

# docker exec -it 02_gradio_frontend pytest -v
docker exec -it sd_txt2img_app pytest -v
# docker exec -it 04_SD_img2img_app pytest -v
# docker exec -it 05_SD_inpainting_app pytest -v
