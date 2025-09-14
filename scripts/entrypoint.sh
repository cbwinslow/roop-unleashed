#!/bin/bash
set -e

# Create models directory
mkdir -p /app/models

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