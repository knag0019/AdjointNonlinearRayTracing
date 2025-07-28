# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode
ARG UID=1000
ARG GID=1000

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git sudo \
    python3.8 python3.8-dev python3-pip python3.8-distutils \
    libpython3.8-dev ca-certificates wget curl \
 && rm -rf /var/lib/apt/lists/*

# ---- Make python3.8 default ----
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    python -m pip install --upgrade pip

# ---- Python packages (install as root to /usr/local) ----
# PyTorch 2.4.1 + cu118 は Python3.8 対応
RUN pip install --no-cache-dir numpy matplotlib tqdm pillow \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118

ENV CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1

# ---- Devcontainer user ----
RUN groupadd -g ${GID} ${USER} && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER}

USER ${USER}
WORKDIR /workspaces

SHELL ["/bin/bash", "-lc"]

# PATH 調整（念のため）
RUN echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
