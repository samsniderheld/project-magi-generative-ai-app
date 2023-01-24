#!/bin/bash
echo "killing old docker processes"
docker kill $(docker ps -q)
docker rm $(docker ps -a -q)

echo "building and running docker containers"
docker-compose up --build -d



while getopts "t" option
do
        case $option in 
                t)
                         echo "running tests"
                         # docker exec -it 02_gradio_frontend pytest -v
                        docker exec -it sd_txt2img_app pytest -v
                        docker exec -it sd_img2img_app pytest -v
                        docker exec -it sd_inpainting_app pytest -v
                        exit
                        ;;
        esac
done