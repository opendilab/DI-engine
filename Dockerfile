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
    && python3 -m pip install --no-cache-dir .[common_env]

# install ROMs with rar
RUN curl -sL https://www.rarlab.com/rar/rarlinux-x64-6.0.1.tar.gz | tar -zxvf - \
    && cd rar && make \
    && cd .. && rm -rf rar \
    && mkdir roms && cd roms \ 
    && curl -O http://www.atarimania.com/roms/Roms.rar && unrar e Roms.rar \
    && python -m atari_py.import_roms . \
    && cd .. && rm -rf roms \
    && rm /usr/local/bin/rar /usr/local/bin/unrar /etc/rarfiles.lst /usr/local/lib/default.sfx
