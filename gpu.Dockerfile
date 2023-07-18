FROM nvidia/cuda:11.2.2-base-ubuntu20.04 AS base_image

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

##Docker config stolen from 
#https://github.com/aws/deep-learning-containers/blob/master/tensorflow/training/docker/2.11/py3/cu112/Dockerfile.gpu

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

FROM base_image AS common

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

# prevent stopping by user interaction
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set environment variables for MKL
# For more about MKL with TensorFlow see:
# https://www.tensorflow.org/performance/performance_guide#tensorflow_with_intel%C2%AE_mkl_dnn
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV KMP_BLOCKTIME=1
ENV KMP_SETTINGS=0
ENV RDMAV_FORK_SAFE=1

ARG PYTHON=python3.9
ARG PIP=pip3
ARG PYTHON_VERSION=3.9.10

ARG OPEN_MPI_PATH=/opt/amazon/openmpi
ARG EFA_PATH=/opt/amazon/efa
ARG EFA_VERSION=1.17.2
ARG OMPI_VERSION=4.1.1
ARG BRANCH_OFI=1.4.0-aws

#Tensorflow and cuda compatibility
#https://www.tensorflow.org/install/source#gpu
ARG CUDA=11.2
ARG CUDA_DASH=11-2
ARG CUDNN=8.2.4.15-1

ARG NCCL_VERSION=2.13.4

# To be passed to ec2 and sagemaker stages
ENV PYTHON=${PYTHON}
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV PIP=${PIP}

RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated --allow-downgrades  --allow-change-held-packages \
   ca-certificates \
   cuda-command-line-tools-${CUDA_DASH} \
   cuda-cudart-dev-${CUDA_DASH} \
   libcufft-dev-${CUDA_DASH} \
   libcurand-dev-${CUDA_DASH} \
   libcusolver-dev-${CUDA_DASH} \
   libcusparse-dev-${CUDA_DASH} \
   curl \
   emacs \
   libcudnn8=${CUDNN}+cuda11.4 \
   libgomp1 \
   libfreetype6-dev \
   libhdf5-serial-dev \
   liblzma-dev \
   libpng-dev \
   libtemplate-perl \
   libzmq3-dev \
   hwloc \
   git \
   unzip \
   wget \
   libtool \
   vim \
   libssl1.1 \
   openssl \
   build-essential \
   openssh-client \
   openssh-server \
   zlib1g-dev \
   # Install dependent library for OpenCV
   libgtk2.0-dev \
   jq \
 && apt-get update \
 && apt-get install -y --no-install-recommends --allow-unauthenticated --allow-change-held-packages \
   libcublas-dev-${CUDA_DASH} \
   libcublas-${CUDA_DASH} \
   # The 'apt-get install' of nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0
   # adds a new list which contains libnvinfer library, so it needs another
   # 'apt-get update' to retrieve that list before it can actually install the
   # library.
   # We don't install libnvinfer-dev since we don't need to build against TensorRT,
   # and libnvinfer4 doesn't contain libnvinfer.a static library.
   # nvinfer-runtime-trt-repo doesn't have a 1804-cuda10.1 version yet. see:
   # https://developer.download.nvidia.cn/compute/machine-learning/repos/ubuntu1804/x86_64/
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /var/run/sshd

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    python3-dev \
    virtualenv \
    swig
    
RUN python3 -m pip --no-cache-dir install \
    Pillow \
    h5py \
    ipdb \
    matplotlib \
    graphviz \
    hist \
    jupyter \
    keras_applications \
    keras_preprocessing \
    mock \
    networkx \
    numba \
    numpy \
    pandas\
    parsl \
    seaborn \
    scipy \
    scikit-learn \
    tqdm \
    future \
    portpicker \
    uproot     \
    uproot4    \
    vtk \
    enum34


# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION=2.11.0
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

#Install graph nets package
RUN python3 -m pip --no-cache-dir install graph_nets "tensorflow_gpu>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability

#Install tendorflow optimisation package
RUN python3 -m pip --no-cache-dir install --upgrade tensorflow-model-optimization

#Install specific branch of hls4ml
RUN git clone https://github.com/fastmachinelearning/hls4ml \
    && cd hls4ml \
    && git checkout main \
    && pip install . \
    && python setup.py install

#Install qkeras
RUN git clone --branch=master https://github.com/google/qkeras.git google/qkeras \
    && cd google/qkeras \
    && pip install . \
    && python setup.py install


RUN pip install protobuf==3.19.*

RUN cd /lib/x86_64-linux-gnu \
    && ln -s libtinfo.so.6 libtinfo.so.5

#ADD symlinks for vivado to work
COPY bashrcFiles/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc