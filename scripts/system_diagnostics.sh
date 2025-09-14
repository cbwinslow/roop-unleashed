#!/usr/bin/env bash

# Roop-Unleashed System Diagnostics and Troubleshooting Tool
# Comprehensive system analysis for optimal performance

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly LOG_FILE="/tmp/roop_diagnostics_$(date +%Y%m%d_%H%M%S).log"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

# System information collection
collect_system_info() {
    log_info "Collecting system information..."
    
    echo "=== SYSTEM INFORMATION ===" >> "$LOG_FILE"
    echo "Date: $(date)" >> "$LOG_FILE"
    echo "Hostname: $(hostname)" >> "$LOG_FILE"
    echo "Kernel: $(uname -a)" >> "$LOG_FILE"
    echo "Architecture: $(uname -m)" >> "$LOG_FILE"
    
    if [ -f /etc/os-release ]; then
        echo "Distribution:" >> "$LOG_FILE"
        cat /etc/os-release >> "$LOG_FILE"
    fi
    
    echo "CPU Information:" >> "$LOG_FILE"
    if [ -f /proc/cpuinfo ]; then
        grep -E "(processor|model name|cpu cores|flags)" /proc/cpuinfo | head -20 >> "$LOG_FILE"
    fi
    
    echo "Memory Information:" >> "$LOG_FILE"
    if [ -f /proc/meminfo ]; then
        grep -E "(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree)" /proc/meminfo >> "$LOG_FILE"
    fi
    
    echo "Disk Space:" >> "$LOG_FILE"
    df -h >> "$LOG_FILE" 2>/dev/null || true
    
    log_success "System information collected"
}

# GPU diagnostics
check_gpu_status() {
    log_info "Checking GPU status..."
    
    echo "=== GPU DIAGNOSTICS ===" >> "$LOG_FILE"
    
    # NVIDIA GPU check
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected:" >> "$LOG_FILE"
        nvidia-smi >> "$LOG_FILE" 2>&1 || echo "nvidia-smi failed" >> "$LOG_FILE"
        
        # Check CUDA installation
        if command -v nvcc &> /dev/null; then
            echo "CUDA Version:" >> "$LOG_FILE"
            nvcc --version >> "$LOG_FILE" 2>&1
        else
            echo "CUDA compiler not found" >> "$LOG_FILE"
        fi
        
        log_success "NVIDIA GPU detected and working"
    else
        log_warn "No NVIDIA GPU or nvidia-smi not found"
    fi
    
    # AMD GPU check
    if command -v rocm-smi &> /dev/null; then
        echo "AMD GPU with ROCm detected:" >> "$LOG_FILE"
        rocm-smi >> "$LOG_FILE" 2>&1 || echo "rocm-smi failed" >> "$LOG_FILE"
        log_success "AMD GPU with ROCm detected"
    else
        # Check for AMD GPU without ROCm
        if lspci | grep -i amd | grep -i vga &> /dev/null; then
            log_warn "AMD GPU detected but ROCm not installed"
            echo "AMD GPU detected (no ROCm):" >> "$LOG_FILE"
            lspci | grep -i amd | grep -i vga >> "$LOG_FILE"
        fi
    fi
    
    # Intel GPU check
    if lspci | grep -E "(VGA|Display)" | grep -i intel &> /dev/null; then
        log_info "Intel GPU detected"
        echo "Intel GPU:" >> "$LOG_FILE"
        lspci | grep -E "(VGA|Display)" | grep -i intel >> "$LOG_FILE"
    fi
    
    # List all GPU devices
    echo "All GPU devices:" >> "$LOG_FILE"
    lspci | grep -E "(VGA|Display|3D)" >> "$LOG_FILE" 2>/dev/null || true
}

# Python environment diagnostics
check_python_environment() {
    log_info "Checking Python environment..."
    
    echo "=== PYTHON ENVIRONMENT ===" >> "$LOG_FILE"
    
    # Check Python version
    if command -v python &> /dev/null; then
        echo "Python version:" >> "$LOG_FILE"
        python --version >> "$LOG_FILE" 2>&1
        
        echo "Python executable path:" >> "$LOG_FILE"
        which python >> "$LOG_FILE" 2>&1
        
        # Check if we're in a virtual environment
        if [[ -n "${VIRTUAL_ENV:-}" ]]; then
            echo "Virtual environment: $VIRTUAL_ENV" >> "$LOG_FILE"
            log_success "Python virtual environment detected"
        elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
            echo "Conda environment: $CONDA_DEFAULT_ENV" >> "$LOG_FILE"
            log_success "Conda environment detected"
        else
            log_warn "No Python virtual environment detected"
        fi
        
        # Check key packages
        local packages=("torch" "torchvision" "opencv-python" "numpy" "insightface" "gradio" "onnxruntime")
        
        echo "Checking key packages:" >> "$LOG_FILE"
        for package in "${packages[@]}"; do
            if python -c "import $package; print(f'$package: {$package.__version__}')" >> "$LOG_FILE" 2>&1; then
                log_success "$package installed"
            else
                log_error "$package not installed or has issues"
            fi
        done
        
        # PyTorch CUDA check
        if python -c "import torch" &> /dev/null; then
            echo "PyTorch CUDA availability:" >> "$LOG_FILE"
            python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"None\"}')" >> "$LOG_FILE" 2>&1
        fi
        
    else
        log_error "Python not found in PATH"
    fi
}

# Check installation integrity
check_installation() {
    log_info "Checking Roop-Unleashed installation..."
    
    echo "=== INSTALLATION CHECK ===" >> "$LOG_FILE"
    
    # Check main files
    local required_files=("run.py" "requirements.txt" "roop/__init__.py")
    
    for file in "${required_files[@]}"; do
        if [ -f "$REPO_DIR/$file" ]; then
            log_success "$file found"
        else
            log_error "$file missing"
        fi
    done
    
    # Check models directory
    if [ -d "$REPO_DIR/models" ]; then
        log_success "Models directory exists"
        echo "Models directory contents:" >> "$LOG_FILE"
        ls -la "$REPO_DIR/models/" >> "$LOG_FILE" 2>/dev/null || true
    else
        log_warn "Models directory not found"
    fi
    
    # Check conda/virtual environment
    local conda_env="$REPO_DIR/installer/installer_files/env"
    if [ -d "$conda_env" ]; then
        log_success "Conda environment found"
        echo "Conda environment path: $conda_env" >> "$LOG_FILE"
    else
        log_warn "Conda environment not found"
    fi
}

# Performance benchmarks
run_performance_tests() {
    log_info "Running performance tests..."
    
    echo "=== PERFORMANCE TESTS ===" >> "$LOG_FILE"
    
    if command -v python &> /dev/null && python -c "import torch" &> /dev/null; then
        # CPU performance test
        echo "CPU Performance Test:" >> "$LOG_FILE"
        python -c "
import time
import torch
import numpy as np

# CPU matrix multiplication test
size = 1000
a = torch.randn(size, size)
b = torch.randn(size, size)

start_time = time.time()
c = torch.matmul(a, b)
cpu_time = time.time() - start_time

print(f'CPU matrix multiplication ({size}x{size}): {cpu_time:.3f} seconds')
" >> "$LOG_FILE" 2>&1
        
        # GPU performance test (if available)
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" &> /dev/null; then
            echo "GPU Performance Test:" >> "$LOG_FILE"
            python -c "
import time
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    size = 1000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    torch.matmul(a, b)
    torch.cuda.synchronize()
    
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f'GPU matrix multiplication ({size}x{size}): {gpu_time:.3f} seconds')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('GPU not available for testing')
" >> "$LOG_FILE" 2>&1
        fi
        
        log_success "Performance tests completed"
    else
        log_warn "Cannot run performance tests (Python/PyTorch not available)"
    fi
}

# Network connectivity check
check_network() {
    log_info "Checking network connectivity..."
    
    echo "=== NETWORK CHECK ===" >> "$LOG_FILE"
    
    # Test key URLs
    local urls=(
        "https://github.com"
        "https://huggingface.co"
        "https://download.pytorch.org"
        "https://repo.anaconda.com"
    )
    
    for url in "${urls[@]}"; do
        if curl -s --connect-timeout 10 "$url" > /dev/null; then
            echo "✅ $url - Accessible" >> "$LOG_FILE"
            log_success "$url accessible"
        else
            echo "❌ $url - Not accessible" >> "$LOG_FILE"
            log_error "$url not accessible"
        fi
    done
}

# Disk space analysis
analyze_disk_usage() {
    log_info "Analyzing disk usage..."
    
    echo "=== DISK USAGE ANALYSIS ===" >> "$LOG_FILE"
    
    # Overall disk usage
    echo "Disk space usage:" >> "$LOG_FILE"
    df -h >> "$LOG_FILE" 2>/dev/null || true
    
    # Specific directory analysis
    if [ -d "$REPO_DIR" ]; then
        echo "Roop-Unleashed directory usage:" >> "$LOG_FILE"
        du -h -d 2 "$REPO_DIR" 2>/dev/null | sort -hr >> "$LOG_FILE" || true
    fi
    
    # Temporary files
    echo "Temporary directory usage:" >> "$LOG_FILE"
    du -h /tmp 2>/dev/null | tail -1 >> "$LOG_FILE" || true
    
    log_success "Disk usage analysis completed"
}

# Generate system report
generate_report() {
    log_info "Generating comprehensive system report..."
    
    collect_system_info
    check_gpu_status
    check_python_environment
    check_installation
    check_network
    analyze_disk_usage
    
    if [ "${RUN_PERFORMANCE_TESTS:-false}" = "true" ]; then
        run_performance_tests
    fi
    
    echo ""
    log_success "Diagnostic report generated: $LOG_FILE"
    echo ""
    echo "Report summary:"
    echo "==============="
    
    # Count issues
    local errors=$(grep -c "\[ERROR\]" "$LOG_FILE" || echo "0")
    local warnings=$(grep -c "\[WARN\]" "$LOG_FILE" || echo "0")
    local successes=$(grep -c "\[SUCCESS\]" "$LOG_FILE" || echo "0")
    
    echo "✅ Successes: $successes"
    echo "⚠️  Warnings: $warnings"
    echo "❌ Errors: $errors"
    
    if [ "$errors" -gt 0 ]; then
        echo ""
        echo "Critical issues found:"
        grep "\[ERROR\]" "$LOG_FILE" | sed 's/.*\[ERROR\]/❌/' || true
    fi
    
    if [ "$warnings" -gt 0 ]; then
        echo ""
        echo "Warnings found:"
        grep "\[WARN\]" "$LOG_FILE" | sed 's/.*\[WARN\]/⚠️ /' || true
    fi
    
    echo ""
    echo "Full report available at: $LOG_FILE"
}

# Quick health check
quick_check() {
    log_info "Running quick health check..."
    
    local issues=0
    
    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found"
        ((issues++))
    fi
    
    # Check key files
    if [ ! -f "$REPO_DIR/run.py" ]; then
        log_error "run.py not found"
        ((issues++))
    fi
    
    # Check GPU (if NVIDIA)
    if command -v nvidia-smi &> /dev/null; then
        if ! nvidia-smi &> /dev/null; then
            log_error "NVIDIA GPU issues detected"
            ((issues++))
        fi
    fi
    
    # Check PyTorch installation
    if command -v python &> /dev/null; then
        if ! python -c "import torch" &> /dev/null; then
            log_error "PyTorch not installed or has issues"
            ((issues++))
        fi
    fi
    
    if [ $issues -eq 0 ]; then
        log_success "Quick health check passed - no critical issues found"
    else
        log_error "Quick health check failed - $issues issues found"
        echo "Run '$0 full-report' for detailed analysis"
    fi
    
    return $issues
}

# Fix common issues
fix_common_issues() {
    log_info "Attempting to fix common issues..."
    
    # Fix permissions
    if [ -d "$REPO_DIR" ]; then
        log_info "Fixing file permissions..."
        find "$REPO_DIR" -name "*.py" -exec chmod +r {} \; 2>/dev/null || true
        find "$REPO_DIR" -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    fi
    
    # Clean temporary files
    log_info "Cleaning temporary files..."
    find /tmp -name "*roop*" -type f -mtime +7 -delete 2>/dev/null || true
    
    # Reset Python cache
    if [ -d "$REPO_DIR" ]; then
        log_info "Clearing Python cache..."
        find "$REPO_DIR" -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true
        find "$REPO_DIR" -name "*.pyc" -delete 2>/dev/null || true
    fi
    
    log_success "Common issue fixes applied"
}

# Show help
show_help() {
    cat << EOF
Roop-Unleashed System Diagnostics Tool

Usage: $0 [COMMAND]

COMMANDS:
    quick-check             Run quick health check
    full-report             Generate comprehensive system report
    gpu-check               Check GPU status only
    python-check            Check Python environment only
    performance-test        Run performance benchmarks
    fix-common              Attempt to fix common issues
    network-check           Test network connectivity
    
EXAMPLES:
    $0 quick-check          # Quick system health check
    $0 full-report          # Complete diagnostic report
    $0 gpu-check            # GPU status only

EOF
}

# Main function
main() {
    case "${1:-quick-check}" in
        quick-check|quick)
            quick_check
            ;;
        full-report|full|report)
            RUN_PERFORMANCE_TESTS=true generate_report
            ;;
        gpu-check|gpu)
            check_gpu_status
            ;;
        python-check|python)
            check_python_environment
            ;;
        performance-test|perf|benchmark)
            run_performance_tests
            ;;
        fix-common|fix)
            fix_common_issues
            ;;
        network-check|network)
            check_network
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"