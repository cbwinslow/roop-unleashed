# Docker Setup for Roop-Unleashed

This repository includes comprehensive Docker support for running roop-unleashed with modern GPU drivers, optimized performance, and ComfyUI-inspired features.

## 🚀 Quick Start

The fastest way to get started:

```bash
# Make the start script executable
chmod +x docker-start.sh

# Run the quick start script
./docker-start.sh
```

This script will:
- Auto-detect your GPU type (NVIDIA, AMD, or CPU-only)
- Create necessary directories
- Set up environment variables
- Start the appropriate Docker configuration
- Wait for the service to be ready

## 📋 Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **GPU Drivers** (for GPU support):
  - NVIDIA: CUDA 12.4+ and nvidia-docker2
  - AMD: ROCm 5.7+

### NVIDIA GPU Setup

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### AMD GPU Setup

```bash
# Install ROCm (Ubuntu 22.04)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install rocm-dkms
```

## 🎯 Available Configurations

### 1. Production (NVIDIA GPU) - Default
```bash
docker-compose up -d roop-unleashed
```
- Latest CUDA 12.4 runtime
- TensorRT optimization support
- Production-optimized settings
- Available at: http://localhost:7860

### 2. Development Environment
```bash
docker-compose --profile dev up -d roop-dev
```
- Includes Jupyter Lab (port 8888)
- TensorBoard support (port 6006)
- Development tools and profiling
- Live code reload
- Debugging utilities

### 3. CPU-Only Mode
```bash
docker-compose --profile cpu up -d roop-cpu
```
- Optimized for CPU processing
- Intel MKL optimizations
- No GPU dependencies
- Available at: http://localhost:7862

### 4. AMD ROCm Support
```bash
docker-compose --profile rocm up -d roop-rocm
```
- ROCm 5.7 support
- AMD GPU optimization
- Available at: http://localhost:7863

### 5. AI/LLM Enhanced
```bash
docker-compose --profile ai up -d ollama roop-unleashed
```
- Local LLM support via Ollama
- RAG (Retrieval Augmented Generation)
- Natural language processing

### 6. Full Monitoring Stack
```bash
docker-compose --profile monitoring up -d prometheus grafana
```
- Prometheus metrics collection
- Grafana dashboards
- GPU and system monitoring

## ⚙️ Configuration

### Environment Variables

Copy the example environment file and customize:

```bash
cp .env.example .env
# Edit .env with your preferences
```

Key configuration options:

```bash
# GPU Settings
CUDA_VISIBLE_DEVICES=all
ROOP_EXECUTION_PROVIDER=cuda
ENABLE_TENSORRT=false

# Performance
ROOP_MAX_THREADS=4
ROOP_FRAME_BUFFER_SIZE=4
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Enhanced Features
ROOP_ENABLE_ENHANCED_PROCESSING=true
ROOP_TEMPORAL_CONSISTENCY=true
ROOP_ADVANCED_INPAINTING=true
```

### Volume Mapping

The following directories are mounted as volumes:
- `./models` - Model files (auto-downloaded)
- `./output` - Generated output files
- `./temp` - Temporary processing files
- `./logs` - Application logs
- `./knowledge` - RAG knowledge base
- `./rag_vectors` - Vector embeddings

## 🎮 GPU Optimization Features

### NVIDIA Optimizations
- **CUDA 12.4** with cuDNN 8
- **TensorRT** support for inference acceleration
- **Multi-GPU** support
- **Memory optimization** with smart allocation
- **Mixed precision** training (FP16/FP32)

### AMD ROCm Optimizations
- **ROCm 5.7** with optimized PyTorch
- **HIP** acceleration
- **ROCm libraries** (ROCblas, ROCfft, etc.)
- **Multi-device** support

### CPU Optimizations
- **Intel MKL** optimizations
- **Multi-threading** with optimal thread counts
- **Memory alignment** optimizations
- **OpenMP** parallelization

## 🧠 ComfyUI-Inspired Features

The Docker setup includes all ComfyUI-inspired enhancements:

### Advanced Processing Pipeline
- **Temporal Consistency**: Smooth video transitions
- **Advanced Inpainting**: Seamless artifact removal
- **Face Quality Analysis**: Multi-metric assessment
- **Enhanced Blending**: Multiple blending methods

### Workflow Integration
- **Modular Processing**: Plugin-based architecture
- **Quality Filtering**: Automatic face selection
- **Batch Processing**: Efficient multi-image handling
- **Memory Management**: Optimized for large workflows

## 📊 Monitoring and Debugging

### Development Tools
Access development tools when using the dev profile:
- **Jupyter Lab**: http://localhost:8888
- **TensorBoard**: http://localhost:6006
- **Debug Console**: `docker-compose exec roop-dev python3 /app/debug_session.py`

### Profiling
```bash
# CPU profiling
docker-compose exec roop-dev python3 -m cProfile -o /app/profiling/profile.stats run.py

# Memory profiling
docker-compose exec roop-dev python3 -m memory_profiler run.py

# Line profiling
docker-compose exec roop-dev kernprof -l -v run.py
```

### Monitoring
With the monitoring profile enabled:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 🐛 Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check NVIDIA
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi

# Check AMD
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi
```

**Out of memory errors:**
```bash
# Reduce batch size and buffer
export ROOP_FRAME_BUFFER_SIZE=2
export ROOP_MAX_THREADS=2
```

**Permission issues:**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER models output temp logs
```

### Performance Tuning

**For high-end GPUs:**
```bash
export ENABLE_TENSORRT=true
export TENSORRT_PRECISION=fp16
export ROOP_FRAME_BUFFER_SIZE=8
```

**For limited memory:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export ROOP_FRAME_BUFFER_SIZE=2
export ROOP_MAX_THREADS=2
```

## 🛠️ Building Custom Images

### Build all variants:
```bash
# NVIDIA GPU version
docker build -f Dockerfile -t roop-unleashed:nvidia .

# Development version
docker build -f Dockerfile.dev -t roop-unleashed:dev .

# CPU version
docker build -f Dockerfile.cpu -t roop-unleashed:cpu .

# AMD ROCm version
docker build -f Dockerfile.rocm -t roop-unleashed:rocm .
```

### Custom Configuration
Create a custom docker-compose override:
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  roop-unleashed:
    environment:
      - CUSTOM_SETTING=value
    volumes:
      - ./custom_models:/app/custom_models
```

## 📚 Advanced Usage

### Multi-GPU Setup
```bash
export CUDA_VISIBLE_DEVICES=0,1
docker-compose up -d roop-unleashed
```

### Cluster Deployment
```bash
# Use Docker Swarm or Kubernetes
docker stack deploy -c docker-compose.yml roop-stack
```

### Custom Models
```bash
# Mount custom model directory
volumes:
  - ./my_models:/app/models/custom
```

## 🔧 Maintenance

### Update Images
```bash
docker-compose pull
docker-compose up -d --force-recreate
```

### Clean Up
```bash
# Remove unused images
docker image prune -f

# Remove all containers and volumes
docker-compose down -v
docker system prune -f
```

### Backup Data
```bash
# Backup important volumes
tar -czf roop-backup.tar.gz models output logs knowledge
```

## 🤝 Contributing

To contribute to the Docker setup:

1. Test your changes with multiple GPU types
2. Update documentation
3. Ensure backward compatibility
4. Follow security best practices

## 📄 License

Same as the main roop-unleashed project.