FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

WORKDIR /nervex

RUN apt update && apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev -y \
    && apt clean \
    && rm -rf /var/cache/apt/*

ADD setup.py setup.py
ADD app_zoo app_zoo
ADD nervex nervex

ARG http_proxy=http://172.16.1.135:3128
ARG https_proxy=http://172.16.1.135:3128

RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir opencv-python


# docker build -t registry.sensetime.com/cloudnative4ai/nervex:v0.0.1-torch1.4-cuda10.1-cudnn7-devel .