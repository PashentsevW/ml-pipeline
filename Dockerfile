FROM --platform=amd64 ubuntu:18.04

LABEL maintainer="antiguru110894@gmail.com" version="0.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV REMOTE_HOME /root

WORKDIR ${REMOTE_HOME}
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -L -o miniconda.sh \
 && chmod +x miniconda.sh \
 && ${REMOTE_HOME}/miniconda.sh -b -p /opt/conda \
 && rm -f miniconda.sh 

ENV PATH /opt/conda/bin:$PATH

RUN activate base \
 && conda install python=3.8.5 -y \
 && pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn==0.23.2 \
    xgboost \
    lightgbm \
    catboost \
    hyperopt \
    scikit-optimize \
    optuna \
    --user