#!/bin/bash
# Docker configuration validation script

set -e

echo "🔍 Validating Docker Configuration for Roop-Unleashed"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Success/failure tracking
CHECKS_PASSED=0
TOTAL_CHECKS=0

check_result() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

# Check Docker installation
echo "📋 Checking Prerequisites..."
docker --version > /dev/null 2>&1
check_result $? "Docker is installed"

docker compose version > /dev/null 2>&1
check_result $? "Docker Compose is available"

# Check if NVIDIA Docker support is available
if command -v nvidia-smi > /dev/null 2>&1; then
    docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1
    check_result $? "NVIDIA GPU support available"
else
    echo -e "${YELLOW}⚠️  NVIDIA GPU not detected (will use CPU mode)${NC}"
fi

# Validate Docker Compose files
echo ""
echo "📋 Validating Docker Compose Configuration..."

docker compose -f docker-compose.yml config > /dev/null 2>&1
check_result $? "docker-compose.yml syntax is valid"

# Check if all Dockerfiles exist
for dockerfile in Dockerfile Dockerfile.dev Dockerfile.cpu Dockerfile.rocm; do
    if [ -f "$dockerfile" ]; then
        echo -e "${GREEN}✅ $dockerfile exists${NC}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo -e "${RED}❌ $dockerfile missing${NC}"
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
done

# Check if environment files exist
if [ -f ".env.example" ]; then
    echo -e "${GREEN}✅ .env.example exists${NC}"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "${RED}❌ .env.example missing${NC}"
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

# Check if essential directories can be created
echo ""
echo "📋 Checking Directory Structure..."
for dir in models output temp logs knowledge rag_vectors; do
    mkdir -p "$dir" 2>/dev/null
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✅ $dir directory ready${NC}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo -e "${RED}❌ Cannot create $dir directory${NC}"
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
done

# Check port availability
echo ""
echo "📋 Checking Port Availability..."
for port in 7860 7861 7862 7863 8888 6006; do
    if ! netstat -ln 2>/dev/null | grep ":$port " > /dev/null; then
        echo -e "${GREEN}✅ Port $port is available${NC}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo -e "${YELLOW}⚠️  Port $port is in use${NC}"
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
done

# Test Docker build (dry run)
echo ""
echo "📋 Testing Docker Configuration..."

# Test if we can validate the main Dockerfile
docker build --target=base -f Dockerfile . --dry-run > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Main Dockerfile build configuration is valid${NC}"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "${YELLOW}⚠️  Main Dockerfile may have build issues${NC}"
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

# Summary
echo ""
echo "📊 Validation Summary"
echo "===================="
echo -e "Checks passed: ${GREEN}$CHECKS_PASSED${NC} / $TOTAL_CHECKS"

if [ $CHECKS_PASSED -eq $TOTAL_CHECKS ]; then
    echo -e "${GREEN}🎉 All checks passed! Your Docker setup is ready.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Copy .env.example to .env and customize if needed"
    echo "2. Run ./docker-start.sh to start the application"
    exit 0
elif [ $CHECKS_PASSED -gt $((TOTAL_CHECKS * 7 / 10)) ]; then
    echo -e "${YELLOW}⚠️  Most checks passed. Review warnings above.${NC}"
    echo ""
    echo "You can likely proceed with:"
    echo "1. ./docker-start.sh"
    exit 0
else
    echo -e "${RED}❌ Several issues detected. Please fix them before proceeding.${NC}"
    echo ""
    echo "Common fixes:"
    echo "1. Install Docker and Docker Compose"
    echo "2. Install NVIDIA Container Toolkit for GPU support"
    echo "3. Check file permissions"
    exit 1
fi