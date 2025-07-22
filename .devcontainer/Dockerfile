# CPU-only version for GitHub Codespaces (.devcontainer/Dockerfile)
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install PyTorch (CPU version) from dedicated index
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies from standard PyPI
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    tqdm \
    Pillow

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Initialize and update submodules
RUN git submodule update --init --recursive

# Create build directory and configure cmake (without CUDA)
RUN mkdir -p build && cd build && \
    cmake .. \
    -DENOKI_CUDA=0 \
    -DENOKI_AUTODIFF=1 \
    -DENOKI_PYTHON=1 && \
    make && \
    cd ..

# Create setpath.sh if it doesn't exist
RUN if [ ! -f setpath.sh ]; then \
        echo '#!/bin/bash' > setpath.sh && \
        echo 'export PYTHONPATH="${PYTHONPATH}:/workspace/build"' >> setpath.sh && \
        chmod +x setpath.sh; \
    fi

# Source the path setup
RUN echo "source /workspace/setpath.sh" >> ~/.bashrc

# Set the default command
CMD ["bash"]
