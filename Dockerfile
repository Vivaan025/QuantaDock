# Multi-stage Docker build for ML model optimization
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS base

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    liblapack-dev \
    libeigen3-dev \
    libfftw3-dev \
    libsndfile1-dev \
    libopencv-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
RUN pip install \
    transformers \
    diffusers \
    accelerate \
    bitsandbytes \
    optimum \
    auto-gptq \
    sentence-transformers \
    librosa \
    soundfile \
    datasets \
    evaluate \
    wandb \
    tensorboard \
    jupyter \
    ipywidgets \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scipy \
    scikit-learn

# Install optimization libraries
RUN pip install \
    tensorrt \
    onnx \
    onnxruntime-gpu \
    openvino \
    torch-tensorrt \
    triton

# Install quantization libraries
RUN pip install \
    quanto \
    llm-int8 \
    peft \
    lora

# Install audio processing libraries
RUN pip install \
    whisper \
    espnet \
    fairseq \
    speechbrain

# Create working directory
WORKDIR /workspace

# Copy optimization scripts
COPY scripts/ /workspace/scripts/
COPY configs/ /workspace/configs/
COPY models/ /workspace/models/

# Set up Jupyter
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose ports
EXPOSE 8888 6006

# Default command
CMD ["bash"]
