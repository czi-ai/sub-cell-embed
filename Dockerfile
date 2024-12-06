FROM mosaicml/pytorch:2.1.2_cu121-python3.10-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --fix-missing \
        curl \
        openssh-client \
        git \
        libgl1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        python3-opencv 
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir \
    omegaconf \
    opencv-python-headless \
    opencv-python \
    colorcet==3.0.1 \
    torchmetrics==1.3.0 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    seaborn==0.13.1 \
    mosaicml-streaming==0.7.3 \
    transformers==4.36.2 \
    wandb==0.16.2 \
    umap-learn \
    scikit-image \
    scipy \
    timm \
    einops \
    awscli 
RUN git clone https://github.com/mosaicml/composer.git
RUN cd composer && pip3 install -e ".[streaming]"