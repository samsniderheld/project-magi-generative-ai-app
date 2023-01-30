# Select base image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Update image
RUN apt-get update && apt-get install -yqq git

# Change working directory
RUN mkdir /app
WORKDIR /app

RUN apt-get install gcc -yqq

RUN apt install wget
# Copy the requirements.txt and install the python packages
COPY requirements.txt .

RUN pip install --upgrade pip  

# Install cython to avoid pycocotools error
RUN pip install cython

# install torch with GPU support for example:
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install -r requirements.txt

# install clip-interrogator stuff
RUN pip install open_clip_torch
RUN pip install -e git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip
#RUN pip install clip-interrogator==0.3.5

# Copy all the content of current directory to working directory
COPY . .

RUN python preprocessing.py

CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 main:app
