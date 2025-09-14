#!/bin/bash
set -e

# Create models directory
mkdir -p /app/models

# Set up CPU-optimized environment
export ROOP_EXECUTION_PROVIDER=cpu
export ROOP_FORCE_CPU=true
export ROOP_EXECUTION_THREADS=${ROOP_EXECUTION_THREADS:-8}
export ROOP_MAX_THREADS=${ROOP_MAX_THREADS:-8}

# CPU optimization settings
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

# Intel optimizations (if available)
export KMP_BLOCKTIME=0
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

# Memory optimization
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Application settings optimized for CPU
export ROOP_FRAME_BUFFER_SIZE=2
export ROOP_MEMORY_LIMIT=0
export ROOP_BATCH_SIZE=1

echo "CPU-optimized environment ready!"

# Exec the command
exec "$@"