#!/bin/bash
# Quick start script for roop-unleashed with Docker

set -e

echo "🚀 Roop-Unleashed Docker Quick Start"
echo "===================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Determine the compose command
COMPOSE_CMD="docker compose"
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models output temp logs knowledge rag_vectors

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "✅ .env file created. Please review and modify as needed."
fi

# Function to detect GPU type
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "nvidia"
    elif lspci | grep -i amd | grep -i vga &> /dev/null; then
        echo "amd"
    else
        echo "cpu"
    fi
}

# Detect GPU and choose appropriate profile
GPU_TYPE=$(detect_gpu)
echo "🔍 Detected GPU type: $GPU_TYPE"

case $GPU_TYPE in
    "nvidia")
        echo "🎮 Starting with NVIDIA GPU support..."
        $COMPOSE_CMD up -d roop-unleashed
        ;;
    "amd")
        echo "🎮 Starting with AMD ROCm support..."
        $COMPOSE_CMD --profile rocm up -d roop-rocm
        ;;
    "cpu")
        echo "💻 Starting with CPU-only mode..."
        $COMPOSE_CMD --profile cpu up -d roop-cpu
        ;;
esac

echo ""
echo "✅ Roop-Unleashed is starting up!"
echo ""
echo "📱 Access the application at: http://localhost:7860"
echo "📊 View logs with: $COMPOSE_CMD logs -f"
echo "🛑 Stop with: $COMPOSE_CMD down"
echo ""
echo "🎯 To enable additional features:"
echo "   - AI/LLM: $COMPOSE_CMD --profile ai up -d ollama"
echo "   - Monitoring: $COMPOSE_CMD --profile monitoring up -d prometheus grafana"
echo "   - Development: $COMPOSE_CMD --profile dev up -d roop-dev"
echo ""

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:7860/ &> /dev/null; then
        echo "✅ Service is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Service may still be starting up. Check logs if needed."
    fi
    sleep 2
done

echo "🎉 Setup complete! Happy face swapping!"