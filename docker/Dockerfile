FROM continuumio/miniconda3

RUN apt-get update -y
RUN apt-get install -y vim
WORKDIR test
RUN git clone https://github.com/funcx-faas/molecular-design
WORKDIR molecular-design

RUN conda env create -f environment.yml

SHELL ["conda", "run", "--no-capture-output", "-n", "myenv", "/bin/bash", "-c"]