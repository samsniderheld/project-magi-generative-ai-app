# Select base image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Update image
RUN apt-get update && apt-get install -yqq git ffmpeg libsm6 libxext6

# Change working directory
RUN mkdir /app
WORKDIR /app

# Copy the requirements.txt and install the python packages
COPY requirements.txt .
RUN pip install --upgrade pip  
RUN pip install -r requirements.txt

# Copy all the content of current directory to working directory
COPY . .


CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 main:app
