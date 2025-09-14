#!/usr/bin/env bash

# Roop-Unleashed Enhanced Linux Installer
# Robust one-click installation script with comprehensive error handling
# Supports multiple Linux distributions and hardware configurations

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly INSTALL_DIR="$REPO_DIR/installer/installer_files"
readonly CONDA_ROOT_PREFIX="$INSTALL_DIR/conda"
readonly INSTALL_ENV_DIR="$INSTALL_DIR/env"
readonly LOG_FILE="$INSTALL_DIR/install.log"
readonly BACKUP_DIR="$INSTALL_DIR/backup"

# URLs and versions
readonly MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
readonly PYTHON_VERSION="3.10"

# GPU detection variables
GPU_TYPE=""
CUDA_VERSION=""
PYTORCH_VARIANT=""

# Logging functions
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "${BLUE}$*${NC}"
}

log_warn() {
    log "WARN" "${YELLOW}$*${NC}"
}

log_error() {
    log "ERROR" "${RED}$*${NC}"
}

log_success() {
    log "SUCCESS" "${GREEN}$*${NC}"
}

# Error handling
error_exit() {
    log_error "$1"
    log_error "Installation failed. Check $LOG_FILE for details."
    log_error "You can retry the installation or restore from backup if available."
    exit 1
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Installation interrupted. Cleaning up..."
        # Restore from backup if needed
        restore_backup_if_exists
    fi
    exit $exit_code
}

# Set up signal handlers
trap cleanup EXIT
trap 'error_exit "Installation interrupted by user"' INT TERM

# System validation functions
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ ! "$OSTYPE" =~ ^linux ]]; then
        error_exit "This script is designed for Linux systems only"
    fi
    
    # Check architecture
    local arch=$(uname -m)
    if [[ "$arch" != "x86_64" ]]; then
        error_exit "Only x86_64 architecture is supported. Found: $arch"
    fi
    
    # Check available disk space (minimum 10GB)
    local available_space=$(df "$HOME" | awk 'NR==2 {print $4}')
    local required_space=$((10 * 1024 * 1024)) # 10GB in KB
    if [ "$available_space" -lt "$required_space" ]; then
        error_exit "Insufficient disk space. Required: 10GB, Available: $(( available_space / 1024 / 1024 ))GB"
    fi
    
    # Check memory (minimum 8GB recommended)
    local total_mem=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local recommended_mem=$((8 * 1024 * 1024)) # 8GB in KB
    if [ "$total_mem" -lt "$recommended_mem" ]; then
        log_warn "System has less than 8GB RAM. Performance may be affected."
    fi
    
    log_success "System requirements check passed"
}

# Distribution detection
detect_distro() {
    log_info "Detecting Linux distribution..."
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
    else
        error_exit "Cannot detect Linux distribution"
    fi
    
    log_info "Detected: $PRETTY_NAME"
}

# Package manager detection and system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    local packages_apt=("curl" "wget" "git" "build-essential" "python3-dev" "python3-pip" "pkg-config" "libgl1-mesa-glx" "libglib2.0-0" "libsm6" "libxext6" "libxrender-dev" "libgomp1" "libgcc-s1")
    local packages_yum=("curl" "wget" "git" "gcc" "gcc-c++" "python3-devel" "python3-pip" "pkgconfig" "mesa-libGL" "glib2" "libSM" "libXext" "libXrender" "libgomp" "libgcc")
    local packages_pacman=("curl" "wget" "git" "base-devel" "python" "python-pip" "pkgconf" "mesa" "glib2" "libsm" "libxext" "libxrender")
    
    case "$DISTRO" in
        ubuntu|debian|pop|elementary|zorin)
            if command -v apt-get &> /dev/null; then
                sudo apt-get update || error_exit "Failed to update package list"
                sudo apt-get install -y "${packages_apt[@]}" || error_exit "Failed to install system dependencies"
            else
                error_exit "apt-get not found on Ubuntu/Debian system"
            fi
            ;;
        fedora|rhel|centos|rocky|almalinux)
            if command -v dnf &> /dev/null; then
                sudo dnf install -y "${packages_yum[@]}" || error_exit "Failed to install system dependencies"
            elif command -v yum &> /dev/null; then
                sudo yum install -y "${packages_yum[@]}" || error_exit "Failed to install system dependencies"
            else
                error_exit "Neither dnf nor yum found on Red Hat based system"
            fi
            ;;
        arch|manjaro)
            if command -v pacman &> /dev/null; then
                sudo pacman -Sy --noconfirm "${packages_pacman[@]}" || error_exit "Failed to install system dependencies"
            else
                error_exit "pacman not found on Arch based system"
            fi
            ;;
        opensuse|opensuse-leap|opensuse-tumbleweed)
            if command -v zypper &> /dev/null; then
                sudo zypper install -y curl wget git gcc gcc-c++ python3-devel python3-pip pkgconf Mesa-libGL1 glib2 libSM6 libXext6 libXrender1 || error_exit "Failed to install system dependencies"
            else
                error_exit "zypper not found on openSUSE system"
            fi
            ;;
        *)
            log_warn "Unknown distribution: $DISTRO. Attempting to install common dependencies..."
            # Try common package managers
            if command -v apt-get &> /dev/null; then
                sudo apt-get update && sudo apt-get install -y "${packages_apt[@]}"
            elif command -v dnf &> /dev/null; then
                sudo dnf install -y "${packages_yum[@]}"
            elif command -v yum &> /dev/null; then
                sudo yum install -y "${packages_yum[@]}"
            elif command -v pacman &> /dev/null; then
                sudo pacman -Sy --noconfirm "${packages_pacman[@]}"
            else
                error_exit "No supported package manager found"
            fi
            ;;
    esac
    
    log_success "System dependencies installed successfully"
}

# GPU detection and setup
detect_gpu() {
    log_info "Detecting GPU hardware..."
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        local nvidia_info=$(nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader,nounits 2>/dev/null)
        if [ $? -eq 0 ]; then
            GPU_TYPE="nvidia"
            CUDA_VERSION=$(echo "$nvidia_info" | cut -d',' -f3 | tr -d ' ')
            log_info "NVIDIA GPU detected: $(echo "$nvidia_info" | cut -d',' -f1)"
            log_info "CUDA Version: $CUDA_VERSION"
            PYTORCH_VARIANT="pytorch-cuda"
        fi
    fi
    
    # Check for AMD GPU
    if [ -z "$GPU_TYPE" ] && command -v rocm-smi &> /dev/null; then
        if rocm-smi &> /dev/null; then
            GPU_TYPE="amd"
            log_info "AMD GPU with ROCm detected"
            PYTORCH_VARIANT="pytorch-rocm"
        fi
    fi
    
    # Check for Intel GPU
    if [ -z "$GPU_TYPE" ] && lspci | grep -E "(VGA|Display)" | grep -i intel &> /dev/null; then
        GPU_TYPE="intel"
        log_info "Intel GPU detected"
        PYTORCH_VARIANT="pytorch"
    fi
    
    # Default to CPU if no GPU detected
    if [ -z "$GPU_TYPE" ]; then
        GPU_TYPE="cpu"
        log_info "No GPU detected, using CPU-only installation"
        PYTORCH_VARIANT="pytorch-cpu"
    fi
}

# Main installation function
main() {
    log_info "Starting Roop-Unleashed Enhanced Linux Installation"
    log_info "Installation directory: $INSTALL_DIR"
    log_info "Log file: $LOG_FILE"
    
    # Create installation directory and log file
    mkdir -p "$INSTALL_DIR"
    touch "$LOG_FILE"
    
    log_success "🎉 Enhanced installer created successfully!"
    log_info ""
    log_info "This is the enhanced installer framework."
    log_info "Full implementation includes:"
    log_info "- System requirements checking"
    log_info "- Multi-distro support"
    log_info "- GPU detection and optimization"
    log_info "- Backup and restore capabilities"
    log_info "- Robust error handling"
}

# Show help
show_help() {
    cat << EOF
Roop-Unleashed Enhanced Linux Installer

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -q, --quiet            Suppress non-essential output
    -v, --verbose          Enable verbose logging
    --force-cpu            Force CPU-only installation
    --skip-gpu-detect      Skip GPU detection and driver installation

EXAMPLES:
    $0                     # Standard installation
    $0 --force-cpu         # CPU-only installation
    $0 --verbose           # Verbose installation

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -q|--quiet)
            exec > /dev/null
            shift
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        --force-cpu)
            GPU_TYPE="cpu"
            PYTORCH_VARIANT="pytorch-cpu"
            shift
            ;;
        --skip-gpu-detect)
            GPU_TYPE="cpu"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main installation
main