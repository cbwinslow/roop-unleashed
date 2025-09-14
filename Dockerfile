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
    onnxruntime-gpu==1.18.1

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/temp /app/logs /app/output /app/rag_vectors /app/knowledge /tmp/matplotlib && \
    chown -R roop:roop /app && \
    chmod -R 755 /app/models /app/temp /app/logs /app/output && \
    chmod 777 /tmp/matplotlib

# Create model download script
COPY scripts/download_models.py /app/download_models.py

# Make scripts executable
RUN chmod +x /app/download_models.py

# Create entrypoint script
COPY scripts/entrypoint.sh /app/entrypoint.sh

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
CMD ["python3", "run.py"]