# Modern Dockerfile for Roop-Unleashed with latest GPU support
# Multi-stage build for optimized final image
FROM nvidia/cuda:12.4-cudnn8-devel-ubuntu22.04 as builder

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libtbb-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install latest pip and wheel
RUN python3 -m pip install --upgrade pip wheel setuptools

# Install PyTorch 2.4+ with CUDA 12.4 support
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Production stage
FROM nvidia/cuda:12.4-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Performance optimizations
ENV OMP_NUM_THREADS=1
ENV MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages

# Install latest pip
RUN python3 -m pip install --upgrade pip

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r roop && useradd -r -g roop roop

# Copy requirements first for better caching
COPY requirements*.txt ./

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    onnxruntime-gpu==1.18.1 \
    tensorrt==10.0.1 \
    pycuda==2024.1 \
    cupy-cuda12x==13.2.0 \
    nvidia-ml-py==12.535.133 \
    transformers==4.42.0 \
    accelerate==0.32.0 \
    xformers==0.0.27 \
    triton==2.4.0

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/temp /app/logs /app/output /app/rag_vectors /app/knowledge && \
    chown -R roop:roop /app

# Create model download script
RUN cat > /app/download_models.py << 'EOF'
#!/usr/bin/env python3
import os
import urllib.request
import hashlib

models = {
    "inswapper_128.onnx": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        "hash": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
    }
}

def download_model(name, info):
    model_path = f"/app/models/{name}"
    if os.path.exists(model_path):
        print(f"Model {name} already exists")
        return
    
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(info["url"], model_path)
    
    # Verify hash if provided
    if "hash" in info:
        with open(model_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != info["hash"]:
            os.remove(model_path)
            raise ValueError(f"Hash mismatch for {name}")
    
    print(f"Downloaded {name} successfully")

if __name__ == "__main__":
    os.makedirs("/app/models", exist_ok=True)
    for name, info in models.items():
        try:
            download_model(name, info)
        except Exception as e:
            print(f"Failed to download {name}: {e}")
EOF

# Make scripts executable
RUN chmod +x /app/download_models.py

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Download models if they don't exist
python3 /app/download_models.py

# Set up environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}
export ROOP_EXECUTION_PROVIDER=${ROOP_EXECUTION_PROVIDER:-cuda}
export ROOP_EXECUTION_THREADS=${ROOP_EXECUTION_THREADS:-4}
export ROOP_MAX_MEMORY=${ROOP_MAX_MEMORY:-0}

# GPU optimization settings
if [ "$ROOP_EXECUTION_PROVIDER" = "cuda" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    export CUDA_LAUNCH_BLOCKING=0
    export CUDA_CACHE_DISABLE=0
fi

# TensorRT optimization
if [ "$ENABLE_TENSORRT" = "true" ]; then
    export TENSORRT_ENABLED=1
    export TENSORRT_PRECISION=${TENSORRT_PRECISION:-fp16}
    export TENSORRT_WORKSPACE_SIZE=${TENSORRT_WORKSPACE_SIZE:-1024}
fi

# Memory optimization
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Exec the command
exec "$@"
EOF

RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER roop

# Expose default Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python3", "run.py", "--server-name", "0.0.0.0", "--server-port", "7860"]