FROM ubuntu

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y zip unzip
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pandas
RUN apt-get install -y python3-matplotlib
RUN apt-get install -y python3-scipy
RUN apt-get install -y python3-sklearn
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-dev
RUN pip3 install -U virtualenv

#TEST Tensorflow build
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:graphics-drivers/ppa

RUN apt-key adv --fetch-keys \
    http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
RUN apt-get update
#RUN apt-get -y install cuda-9-0

RUN pip3 install -U tensorflow-gpu

CMD ["/bin/bash"]
