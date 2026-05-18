FROM nvidia/cuda:12.6.0-devel-ubuntu24.04 
# changed to 12.6.0 so that python 3.12 works.
# start fromm official CUDA image with ubunti 22.04 and CUDA 12.4.1. so we're building on top of it.
# so instead of starting from a blank state we are building on top of official image.

# setting up environment variables to be non-interactive and send python output to my terminal
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ubuntu 22.04 comes with python 3.10 but we want to use python 3.12 so we need to add the deadsnakes ppa to get python 3.12 and set it as the default python version.
# RUN apt-get update && apt-get install -y software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update

# installing the system dependencies  like git (to clone), manage python and install it 
# curl to download models and build-essential for compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3.12-venv \
    python3.12-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# using a venv to avoid fighthing debian's system pip entirely
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# now pip inside the venv is fully ours -no debian restrictions.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# setting up the gpu mathcore(pytorch and cuda)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 


# adding aitools for workspaces
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    transformers \
    datasets \
    accelerate \
    safetensors \
    qdrant-client==1.12.2 \
    flask

# below this setting up the storage and communication ports.
WORKDIR /workspace

VOLUME ["/workspace", "/models"] 
# to persist data and models outside the container. save progress and models even if the container is stopped or removed.

EXPOSE 8888

CMD ["python"]
