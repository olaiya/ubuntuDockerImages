FROM ubuntu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.6 \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-pandas \
    python3-pip \
    python3-scipy \
    python3-sklearn \
    software-properties-common \
    zip unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install uproot
RUN pip3 install -U virtualenv

#TEST Tensorflow build
#RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:graphics-drivers/ppa

RUN apt-key adv --fetch-keys \
    http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
RUN apt-get update
#RUN apt-get -y install cuda-9-0

RUN pip3 install -U tensorflow-gpu

CMD ["/bin/bash"]
