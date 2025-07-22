# CPU-only version for GitHub Codespaces
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.8 /usr/bin/python

# Install Python dependencies (CPU versions)
RUN pip3 install --no-cache-dir \
    numpy \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
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
