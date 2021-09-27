FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

WORKDIR /ding

RUN apt update \
    && apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev swig curl gcc \g++ make locales -y \
    && apt clean \
    && rm -rf /var/cache/apt/* \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

ENV LANG        en_US.UTF-8
ENV LANGUAGE    en_US:UTF-8
ENV LC_ALL      en_US.UTF-8

ADD setup.py setup.py
ADD dizoo dizoo
ADD ding ding

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir .[fast,common_env] \
    && pip install autorom \
    && AutoROM --accept-license
