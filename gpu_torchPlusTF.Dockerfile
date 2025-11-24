# Use Ubuntu 24.04 as base
FROM ubuntu:24.04
#FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    VENV_PATH=/opt/venv

# Install essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tzdata \
      ca-certificates \
      curl \
      emacs \
      wget \
      git \
      openssl \
      openssh-client \
      openssh-server \
      python3.12 \
      python3.12-venv \
      python3-pip \
      unzip \
      vim \
      zlib1g-dev \
      build-essential \
      && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv $VENV_PATH

# Ensure venv Python & pip are used by default
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip inside venv
RUN pip install --upgrade pip setuptools wheel

#https://www.tensorflow.org/install/source#gpu
# ----------------------------------------------------------------------
# Install TensorFlow (GPU-enabled)
# TensorFlow 2.20 ships with prebuilt CUDA 12.x wheels.
# ----------------------------------------------------------------------
RUN pip install --no-cache-dir tensorflow==2.19.0 \
	tf-keras==2.19.0

# Install PyTorch (CPU only example)
#RUN pip install torch==2.9.1 torchvision torchaudio torchao
# (Optional) for CUDA, replace with the matching wheel:
RUN pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
    Pillow \
    h5py \
    ipdb \
    bokeh

RUN pip install \
    matplotlib \
    graphviz \
    hist \
    jupyter \
    keras_applications

RUN pip install \
    keras_preprocessing \
    mock \
    networkx

RUN pip install \
    numba \
    numpy \
    pandas\
    parsl \
    seaborn \
    scipy \
    scikit-learn

RUN pip install \
    tqdm \
    future \
    portpicker \
    uproot     \
    vtk \
    enum34

RUN pip install executorch \
    onnxruntime \
    onnx \
    onnxscript
    
#Installl onnx2tf and its dependencies   
 RUN pip install  simple_onnx_processing_tools \
    onnx==1.17.0 \
    #nvidia-pyindex \
    onnx2tf \
    onnxruntime==1.18.1 \
    #onnxsim==0.4.33 \
    #onnxsim \
    simple_onnx_processing_tools \
    sne4onnx>=1.0.13 \
    sng4onnx>=1.0.4 \
    ai_edge_litert==1.2.0 \
    protobuf==3.20.3 \
    onnx2tf \
    onnx_graphsurgeon \
    h5py==3.11.0 \
    psutil==5.9.5 \
    ml_dtypes==0.5.1 \
    flatbuffers>=23.5.26

WORKDIR /workspace

# Copy your application if needed
# COPY . /workspace

COPY bashrcFiles/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

#CMD ["python", "--version"]
