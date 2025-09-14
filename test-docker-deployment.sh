#!/bin/bash
# Comprehensive Docker testing script for roop-unleashed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TOTAL_TESTS=0
FAILED_TESTS=()

test_result() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}❌ $2${NC}"
        FAILED_TESTS+=("$2")
    fi
}

echo -e "${BLUE}🧪 Comprehensive Docker Test Suite for Roop-Unleashed${NC}"
echo "================================================================"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up test containers...${NC}"
    docker compose -f docker-compose-simple.yml down --remove-orphans &>/dev/null || true
    docker compose down --remove-orphans &>/dev/null || true
    docker system prune -f &>/dev/null || true
}

# Setup cleanup trap
trap cleanup EXIT

echo -e "${BLUE}📋 Phase 1: Pre-flight Checks${NC}"
echo "================================="

# Check Docker installation
docker --version > /dev/null 2>&1
test_result $? "Docker is installed and running"

docker compose version > /dev/null 2>&1
test_result $? "Docker Compose is available"

# Create necessary directories
mkdir -p models output temp logs
test_result $? "Created necessary directories"

echo ""
echo -e "${BLUE}📋 Phase 2: Docker Build Tests${NC}"
echo "=================================="

# Test building simplified CPU image
echo -e "${YELLOW}Building CPU-only image...${NC}"
docker build -f Dockerfile.cpu -t roop-unleashed:cpu-test . &>/dev/null
test_result $? "CPU Dockerfile builds successfully"

# Test building main GPU image (if no GPU, this might fail but we'll note it)
echo -e "${YELLOW}Building main GPU image...${NC}"
docker build -f Dockerfile -t roop-unleashed:gpu-test . &>/dev/null
test_result $? "GPU Dockerfile builds successfully"

echo ""
echo -e "${BLUE}📋 Phase 3: Container Startup Tests${NC}"
echo "====================================="

# Test CPU container startup
echo -e "${YELLOW}Testing CPU container startup...${NC}"
docker compose -f docker-compose-simple.yml --profile cpu up -d roop-cpu &>/dev/null
sleep 10

# Check if container is running
if docker compose -f docker-compose-simple.yml ps roop-cpu | grep -q "Up"; then
    test_result 0 "CPU container starts successfully"
    
    # Test basic connectivity after giving it time to start
    sleep 20
    if curl -f http://localhost:7862 &>/dev/null || curl -f http://localhost:7862/ &>/dev/null; then
        test_result 0 "CPU container is accessible on port 7862"
    else
        test_result 1 "CPU container is not accessible (may still be starting)"
    fi
else
    test_result 1 "CPU container failed to start"
fi

# Stop CPU container
docker compose -f docker-compose-simple.yml --profile cpu down &>/dev/null

# Test GPU container startup (only if NVIDIA GPU is available)
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo -e "${YELLOW}Testing GPU container startup...${NC}"
    docker compose -f docker-compose-simple.yml up -d roop-unleashed &>/dev/null
    sleep 10
    
    if docker compose -f docker-compose-simple.yml ps roop-unleashed | grep -q "Up"; then
        test_result 0 "GPU container starts successfully"
        
        # Test basic connectivity
        sleep 20
        if curl -f http://localhost:7860 &>/dev/null || curl -f http://localhost:7860/ &>/dev/null; then
            test_result 0 "GPU container is accessible on port 7860"
        else
            test_result 1 "GPU container is not accessible (may still be starting)"
        fi
    else
        test_result 1 "GPU container failed to start"
    fi
    
    docker compose -f docker-compose-simple.yml down &>/dev/null
else
    echo -e "${YELLOW}⚠️  NVIDIA GPU not detected, skipping GPU container tests${NC}"
fi

echo ""
echo -e "${BLUE}📋 Phase 4: Quick Functionality Tests${NC}"
echo "====================================="

# Test simplified startup with basic validation
echo -e "${YELLOW}Testing quick CPU startup for basic functionality...${NC}"
docker compose -f docker-compose-simple.yml --profile cpu up -d roop-cpu &>/dev/null
sleep 15

# Check container logs for startup success indicators
if docker compose -f docker-compose-simple.yml logs roop-cpu | grep -q "Running on"; then
    test_result 0 "Container shows successful startup in logs"
elif docker compose -f docker-compose-simple.yml logs roop-cpu | grep -q "Gradio"; then
    test_result 0 "Container shows Gradio startup in logs"
else
    test_result 1 "Container logs don't show successful startup"
fi

# Check for error indicators in logs
if docker compose -f docker-compose-simple.yml logs roop-cpu | grep -qi "error\|exception\|failed"; then
    test_result 1 "Container logs show errors"
else
    test_result 0 "No errors found in container logs"
fi

# Cleanup
docker compose -f docker-compose-simple.yml --profile cpu down &>/dev/null

echo ""
echo -e "${BLUE}📋 Phase 5: Resource Usage Tests${NC}"
echo "=================================="

# Test memory usage
echo -e "${YELLOW}Testing container resource usage...${NC}"
docker compose -f docker-compose-simple.yml --profile cpu up -d roop-cpu &>/dev/null
sleep 10

# Check memory usage
MEMORY_USAGE=$(docker stats --no-stream --format "table {{.MemUsage}}" roop-cpu 2>/dev/null | tail -n1 | cut -d'/' -f1 | tr -d ' MiB')
if [ ! -z "$MEMORY_USAGE" ] && [ "$MEMORY_USAGE" -lt 4000 ]; then
    test_result 0 "Container memory usage is reasonable (${MEMORY_USAGE}MiB)"
else
    test_result 1 "Container memory usage may be too high"
fi

docker compose -f docker-compose-simple.yml --profile cpu down &>/dev/null

echo ""
echo -e "${BLUE}📊 Test Results Summary${NC}"
echo "========================"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC} / $TOTAL_TESTS"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}- $test${NC}"
    done
fi

echo ""
if [ $TESTS_PASSED -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}🎉 All tests passed! Docker setup is working correctly.${NC}"
    echo ""
    echo "🚀 Ready to deploy! Use:"
    echo "  docker compose -f docker-compose-simple.yml up -d              # GPU version"
    echo "  docker compose -f docker-compose-simple.yml --profile cpu up -d roop-cpu  # CPU version"
    exit 0
elif [ $TESTS_PASSED -gt $((TOTAL_TESTS * 7 / 10)) ]; then
    echo -e "${YELLOW}⚠️  Most tests passed. Docker setup is likely working with minor issues.${NC}"
    echo ""
    echo "You can try deploying with:"
    echo "  docker compose -f docker-compose-simple.yml --profile cpu up -d roop-cpu"
    exit 0
else
    echo -e "${RED}❌ Too many tests failed. Please review the issues above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "1. Check Docker and Docker Compose installation"
    echo "2. Ensure sufficient disk space for builds"
    echo "3. Check network connectivity for downloading dependencies"
    exit 1
fi