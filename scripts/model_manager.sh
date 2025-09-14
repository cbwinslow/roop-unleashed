#!/usr/bin/env bash

# Roop-Unleashed Model Management Utility
# Download, manage, and switch between different AI models

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly MODELS_DIR="$REPO_DIR/models"
readonly CONFIG_FILE="$MODELS_DIR/model_config.json"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Model definitions with URLs, checksums, and descriptions
declare -A MODELS=(
    # Face swapping models
    ["inswapper_128.onnx"]="https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx|Face swapping model (128px)|Required"
    ["inswapper_512.onnx"]="https://huggingface.co/deepfakes/inswapper/resolve/main/inswapper_512.onnx|Higher quality face swapping (512px)|Optional"
    
    # Face enhancement models
    ["GFPGANv1.4.pth"]="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth|GFPGAN face enhancement|Optional"
    ["CodeFormer.pth"]="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth|CodeFormer face restoration|Optional"
    ["RestoreFormer.pth"]="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth|RestoreFormer enhancement|Optional"
    
    # Face detection models
    ["buffalo_l.zip"]="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip|InsightFace detection model|Required"
    ["antelopev2.zip"]="https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip|Improved face detection|Optional"
    
    # Experimental models
    ["simswap_512.onnx"]="https://huggingface.co/deepfakes/simswap/resolve/main/simswap_512.onnx|SimSwap alternative model|Experimental"
    ["ghost_unet.onnx"]="https://huggingface.co/countfloyd/deepfake/resolve/main/ghost_unet.onnx|Ghost UNet model|Experimental"
)

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

# Initialize models directory and config
init_models_dir() {
    mkdir -p "$MODELS_DIR"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        cat > "$CONFIG_FILE" << EOF
{
    "installed_models": {},
    "active_models": {
        "face_swapper": "inswapper_128.onnx",
        "face_enhancer": "GFPGANv1.4.pth",
        "face_detector": "buffalo_l"
    },
    "model_paths": {
        "face_swapper": "$MODELS_DIR",
        "face_enhancer": "$MODELS_DIR/enhancers",
        "face_detector": "$MODELS_DIR/detectors"
    }
}
EOF
        log_info "Created model configuration file"
    fi
}

# Show available models
list_models() {
    echo "Available Models:"
    echo "=================="
    
    for model in "${!MODELS[@]}"; do
        IFS='|' read -r url description category <<< "${MODELS[$model]}"
        local status="❌ Not installed"
        
        if [ -f "$MODELS_DIR/$model" ]; then
            status="✅ Installed"
        fi
        
        printf "%-25s %-15s %s\n" "$model" "[$category]" "$description"
        printf "%-25s %s\n" "" "$status"
        echo ""
    done
}

# Download a specific model
download_model() {
    local model_name="$1"
    
    if [[ ! "${MODELS[$model_name]:-}" ]]; then
        log_error "Model '$model_name' not found in registry"
        return 1
    fi
    
    IFS='|' read -r url description category <<< "${MODELS[$model_name]}"
    local model_path="$MODELS_DIR/$model_name"
    
    if [ -f "$model_path" ]; then
        log_warn "Model '$model_name' already exists"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    log_info "Downloading $model_name..."
    log_info "Description: $description"
    log_info "Category: $category"
    
    # Create subdirectories based on model type
    case "$model_name" in
        *enhancer*|*GFPGAN*|*CodeFormer*|*RestoreFormer*)
            mkdir -p "$MODELS_DIR/enhancers"
            model_path="$MODELS_DIR/enhancers/$model_name"
            ;;
        *detector*|*buffalo*|*antelope*)
            mkdir -p "$MODELS_DIR/detectors"
            model_path="$MODELS_DIR/detectors/$model_name"
            ;;
    esac
    
    # Download with progress bar
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$model_path" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$model_path" "$url"
    else
        log_error "Neither wget nor curl found"
        return 1
    fi
    
    # Verify download
    if [ ! -f "$model_path" ] || [ ! -s "$model_path" ]; then
        log_error "Download failed or file is empty"
        rm -f "$model_path"
        return 1
    fi
    
    # Handle zip files
    if [[ "$model_name" == *.zip ]]; then
        log_info "Extracting $model_name..."
        local extract_dir="${model_path%.*}"
        mkdir -p "$extract_dir"
        
        if command -v unzip &> /dev/null; then
            unzip -q "$model_path" -d "$extract_dir"
            rm "$model_path"  # Remove zip after extraction
            log_success "Extracted to $extract_dir"
        else
            log_warn "unzip not found, keeping zip file"
        fi
    fi
    
    log_success "Successfully downloaded $model_name"
}

# Download all required models
download_required() {
    log_info "Downloading all required models..."
    
    for model in "${!MODELS[@]}"; do
        IFS='|' read -r url description category <<< "${MODELS[$model]}"
        
        if [ "$category" = "Required" ]; then
            download_model "$model"
        fi
    done
}

# Download all models
download_all() {
    log_info "Downloading all available models..."
    
    for model in "${!MODELS[@]}"; do
        download_model "$model"
    done
}

# Remove a model
remove_model() {
    local model_name="$1"
    local model_path="$MODELS_DIR/$model_name"
    
    # Check in subdirectories
    for subdir in "enhancers" "detectors"; do
        if [ -f "$MODELS_DIR/$subdir/$model_name" ]; then
            model_path="$MODELS_DIR/$subdir/$model_name"
            break
        fi
    done
    
    if [ ! -f "$model_path" ]; then
        log_error "Model '$model_name' not found"
        return 1
    fi
    
    read -p "Remove $model_name? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$model_path"
        log_success "Removed $model_name"
    fi
}

# Check model integrity
verify_models() {
    log_info "Verifying installed models..."
    
    local verified=0
    local total=0
    
    for model in "${!MODELS[@]}"; do
        ((total++))
        local model_path="$MODELS_DIR/$model"
        
        # Check in subdirectories
        for subdir in "" "enhancers" "detectors"; do
            local check_path="$MODELS_DIR/$subdir/$model"
            if [ -f "$check_path" ]; then
                model_path="$check_path"
                break
            fi
        done
        
        if [ -f "$model_path" ]; then
            local size=$(stat -f%z "$model_path" 2>/dev/null || stat -c%s "$model_path" 2>/dev/null || echo "0")
            if [ "$size" -gt 1000 ]; then  # At least 1KB
                echo "✅ $model - OK (${size} bytes)"
                ((verified++))
            else
                echo "❌ $model - CORRUPTED (${size} bytes)"
            fi
        else
            echo "❌ $model - NOT FOUND"
        fi
    done
    
    log_info "Verified $verified/$total models"
}

# Show disk usage
show_usage() {
    log_info "Model storage usage:"
    
    if [ -d "$MODELS_DIR" ]; then
        du -h "$MODELS_DIR" 2>/dev/null || echo "Unable to calculate disk usage"
    else
        echo "Models directory not found"
    fi
}

# Backup models
backup_models() {
    local backup_dir="$REPO_DIR/model_backups/backup_$(date +%Y%m%d_%H%M%S)"
    
    log_info "Creating backup in $backup_dir..."
    mkdir -p "$backup_dir"
    
    if [ -d "$MODELS_DIR" ]; then
        cp -r "$MODELS_DIR"/* "$backup_dir/" 2>/dev/null || true
        log_success "Backup created: $backup_dir"
    else
        log_warn "No models directory to backup"
    fi
}

# Show help
show_help() {
    cat << EOF
Roop-Unleashed Model Management Utility

Usage: $0 [COMMAND] [OPTIONS]

COMMANDS:
    list                    List all available models
    download <model>        Download a specific model
    download-required       Download all required models
    download-all           Download all available models
    remove <model>         Remove a specific model
    verify                 Verify integrity of installed models
    usage                  Show disk usage of models
    backup                 Create backup of all models
    
EXAMPLES:
    $0 list                           # Show all available models
    $0 download inswapper_128.onnx    # Download specific model
    $0 download-required              # Download required models only
    $0 verify                         # Check model integrity

EOF
}

# Main function
main() {
    init_models_dir
    
    case "${1:-}" in
        list|ls)
            list_models
            ;;
        download)
            if [ $# -lt 2 ]; then
                log_error "Please specify a model name"
                show_help
                exit 1
            fi
            download_model "$2"
            ;;
        download-required)
            download_required
            ;;
        download-all)
            download_all
            ;;
        remove|rm)
            if [ $# -lt 2 ]; then
                log_error "Please specify a model name"
                exit 1
            fi
            remove_model "$2"
            ;;
        verify|check)
            verify_models
            ;;
        usage|du)
            show_usage
            ;;
        backup)
            backup_models
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
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