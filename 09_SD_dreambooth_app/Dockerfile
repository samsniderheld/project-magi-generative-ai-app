# Select base image
#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Update image
RUN apt-get update && apt-get install -yqq git

# Change working directory
RUN mkdir /app
WORKDIR /app

# Copy the requirements.txt and install the python packages
COPY requirements.txt .
RUN pip install --upgrade pip  

RUN pip install -qq git+https://github.com/ShivamShrirao/diffusers
RUN pip install -q -U --pre triton
RUN pip install -r requirements.txt
RUN pip install bitsandbytes-cuda116

#RUN pip install --no-deps -q https://github.com/brian6091/xformers-wheels/releases/download/0.0.15.dev0%2B4c06c79/xformers-0.0.15.dev0+4c06c79.d20221205-cp38-cp38-linux_x86_64.whl

# Copy all the content of current directory to working directory
COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 main:app
