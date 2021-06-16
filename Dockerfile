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

RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir .[common_env]

# install ROMs with rar
RUN curl -sL https://www.rarlab.com/rar/rarlinux-x64-6.0.1.tar.gz | tar -zxvf - \
    && cd rar && make \
    && cd .. && rm -rf rar \
    && mkdir roms && cd roms \ 
    && curl -O http://www.atarimania.com/roms/Roms.rar && unrar e Roms.rar \
    && python -m atari_py.import_roms . \
    && cd .. && rm -rf roms \
    && rm /usr/local/bin/rar /usr/local/bin/unrar /etc/rarfiles.lst /usr/local/lib/default.sfx

# docker build -t registry.sensetime.com/cloudnative4ai/nervex:v0.0.1-torch1.4-cuda10.1-cudnn7-devel .