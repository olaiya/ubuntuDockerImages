#https://github.com/tensorflow/build/blob/fdd023ef684d62b2f76f0ab5ebcffda19d982a21/tensorflow_runtime_dockerfiles/gpu.Dockerfile
#https://gist.github.com/Cyril-Meyer/85ac02355e41437626772fa3741a1935
#https://www.tensorflow.org/install/source#gpu
#https://developer.nvidia.com/cuda-12-3-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
#https://developer.nvidia.com/cudnn-archive
#https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/

FROM nvidia/cuda:12.3.0-base-ubuntu22.04 AS base_image

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda-12.3/lib64/stubs:/usr/local/cuda-12.3/lib64:"

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

ARG PYTHON=python3.9
ARG PIP=pip3
ARG PYTHON_VERSION=3.9.10

#Tensorflow and cuda compatibility
#https://www.tensorflow.org/install/source#gpu
#Check available cuda packages for ubunt here:
#https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/
ARG CUDA=12.3
ARG CUDA_DASH=12-3
ARG CUDNN=8.9.7.29-1

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
   libcudnn8=${CUDNN}+cuda12.2 \
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
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow[and-cuda]
ARG TF_PACKAGE_VERSION=2.16.1
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}


RUN cd /lib/x86_64-linux-gnu \
    && ln -s libtinfo.so.6 libtinfo.so.5

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so

ENV LD_LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/nvidia//nvjitlink/lib:/usr/local/lib/python3.10/dist-packages/nvidia//nccl/lib:/usr/local/lib/python3.10/dist-packages/nvidia//cusparse/lib:/usr/local/lib/python3.10/dist-packages/nvidia//cusolver/lib:/usr/local/lib/python3.10/dist-packages/nvidia//curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia//cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia//cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia//cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia//cuda_nvrtc/lib:/usr/local/lib/python3.10/dist-packages/nvidia//cuda_nvcc/lib:/usr/local/lib/python3.10/dist-packages/nvidia//cuda_cupti/lib:/usr/local/lib/python3.10/dist-packages/nvidia//cublas/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

COPY bashrcFiles/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc