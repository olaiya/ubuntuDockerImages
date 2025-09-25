# Use Ubuntu 24.04 as base
FROM ubuntu:24.04

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

# Install PyTorch (CPU only example)
#RUN pip install torch==2.8.0 torchvision torchaudio torchao
# (Optional) for CUDA, replace with the matching wheel:
RUN pip install torch==2.8.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
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
    vtk \
    enum34

RUN pip install executorch

WORKDIR /workspace

# Copy your application if needed
# COPY . /workspace

CMD ["python", "--version"]
