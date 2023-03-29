FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN apt-get update && apt-get install -y \
ffmpeg libsm6 libxext6 build-essential

WORKDIR /usr/src/edgeyolo
COPY . .
RUN pip install -r requirements.txt
