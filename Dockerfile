FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update -y && \
    apt-get install -y wget libsndfile1-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip install librosa==0.8.0 pytorch-lightning==1.1.4
