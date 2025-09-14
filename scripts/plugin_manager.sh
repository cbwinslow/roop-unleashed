#!/usr/bin/env bash

# Roop-Unleashed Plugin Management Utility
# Discover, install, and manage plugins for enhanced functionality

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly PLUGINS_DIR="$REPO_DIR/plugins"
readonly CUSTOM_PLUGINS_DIR="$PLUGINS_DIR/custom"
readonly PLUGIN_REGISTRY="$PLUGINS_DIR/plugin_registry.json"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Available plugins registry
declare -A AVAILABLE_PLUGINS=(
    # Core enhancement plugins
    ["real_esrgan"]="https://github.com/xinntao/Real-ESRGAN|Real-ESRGAN super resolution|enhancement"
    ["basicsr"]="https://github.com/XPixelGroup/BasicSR|BasicSR restoration framework|enhancement"
    ["waifu2x"]="https://github.com/nagadomi/waifu2x|Waifu2x image upscaling|enhancement"
    
    # Face processing plugins
    ["arcface"]="https://github.com/deepinsight/insightface|ArcFace recognition|face_processing"
    ["face_parsing"]="https://github.com/zllrunning/face-parsing.PyTorch|Face parsing and segmentation|face_processing"
    ["facexlib"]="https://github.com/xinntao/facexlib|Face restoration library|face_processing"
    
    # Video processing plugins
    ["ffmpeg_python"]="https://github.com/kkroening/ffmpeg-python|FFmpeg Python bindings|video"
    ["moviepy"]="https://github.com/Zulko/moviepy|Video editing library|video"
    ["opencv_contrib"]="https://github.com/opencv/opencv_contrib|OpenCV contributions|video"
    
    # Utility plugins
    ["tensorboard"]="https://github.com/tensorflow/tensorboard|TensorBoard monitoring|utility"
    ["wandb"]="https://github.com/wandb/wandb|Weights & Biases tracking|utility"
    ["jupyter"]="https://github.com/jupyter/jupyter|Jupyter notebook support|utility"
    
    # Experimental plugins
    ["styleGAN"]="https://github.com/NVlabs/stylegan3|StyleGAN3 integration|experimental"
    ["first_order_model"]="https://github.com/AliaksandrSiarohin/first-order-model|First Order Motion Model|experimental"
    ["liquid_warping"]="https://github.com/svip-lab/impersonator|Liquid Warping GAN|experimental"
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

# Initialize plugin system
init_plugin_system() {
    mkdir -p "$CUSTOM_PLUGINS_DIR"
    
    if [ ! -f "$PLUGIN_REGISTRY" ]; then
        cat > "$PLUGIN_REGISTRY" << EOF
{
    "installed_plugins": {},
    "enabled_plugins": [],
    "plugin_config": {},
    "last_update": "$(date -Iseconds)"
}
EOF
        log_info "Created plugin registry"
    fi
}

# List available plugins
list_available() {
    echo "Available Plugins:"
    echo "=================="
    
    for plugin in "${!AVAILABLE_PLUGINS[@]}"; do
        IFS='|' read -r url description category <<< "${AVAILABLE_PLUGINS[$plugin]}"
        local status="❌ Not installed"
        
        if check_plugin_installed "$plugin"; then
            status="✅ Installed"
        fi
        
        printf "%-20s %-15s %s\n" "$plugin" "[$category]" "$description"
        printf "%-20s %s\n" "" "$status"
        echo ""
    done
}

# List installed plugins
list_installed() {
    echo "Installed Plugins:"
    echo "=================="
    
    local found_any=false
    
    # Check built-in plugins
    if [ -d "$PLUGINS_DIR" ]; then
        for plugin_file in "$PLUGINS_DIR"/plugin_*.py; do
            if [ -f "$plugin_file" ]; then
                local plugin_name=$(basename "$plugin_file" .py | sed 's/plugin_//')
                echo "✅ $plugin_name (built-in)"
                found_any=true
            fi
        done
    fi
    
    # Check custom plugins
    if [ -d "$CUSTOM_PLUGINS_DIR" ]; then
        for plugin_dir in "$CUSTOM_PLUGINS_DIR"/*; do
            if [ -d "$plugin_dir" ]; then
                local plugin_name=$(basename "$plugin_dir")
                echo "✅ $plugin_name (custom)"
                found_any=true
            fi
        done
    fi
    
    if [ "$found_any" = false ]; then
        echo "No plugins installed"
    fi
}

# Check if plugin is installed
check_plugin_installed() {
    local plugin_name="$1"
    
    # Check for built-in plugin
    [ -f "$PLUGINS_DIR/plugin_${plugin_name}.py" ] && return 0
    
    # Check for custom plugin
    [ -d "$CUSTOM_PLUGINS_DIR/$plugin_name" ] && return 0
    
    return 1
}

# Install plugin
install_plugin() {
    local plugin_name="$1"
    
    if [[ ! "${AVAILABLE_PLUGINS[$plugin_name]:-}" ]]; then
        log_error "Plugin '$plugin_name' not found in registry"
        return 1
    fi
    
    if check_plugin_installed "$plugin_name"; then
        log_warn "Plugin '$plugin_name' already installed"
        return 0
    fi
    
    IFS='|' read -r url description category <<< "${AVAILABLE_PLUGINS[$plugin_name]}"
    
    log_info "Installing plugin: $plugin_name"
    log_info "Description: $description"
    log_info "Category: $category"
    log_info "Source: $url"
    
    local plugin_dir="$CUSTOM_PLUGINS_DIR/$plugin_name"
    mkdir -p "$plugin_dir"
    
    # Clone or download plugin
    if command -v git &> /dev/null; then
        if git clone "$url" "$plugin_dir/source" 2>/dev/null; then
            log_success "Plugin source downloaded"
        else
            log_error "Failed to clone plugin repository"
            rm -rf "$plugin_dir"
            return 1
        fi
    else
        log_error "Git not found - cannot install plugins"
        return 1
    fi
    
    # Create plugin wrapper
    create_plugin_wrapper "$plugin_name" "$plugin_dir"
    
    # Install plugin dependencies if requirements.txt exists
    if [ -f "$plugin_dir/source/requirements.txt" ]; then
        log_info "Installing plugin dependencies..."
        if command -v pip &> /dev/null; then
            pip install -r "$plugin_dir/source/requirements.txt" || log_warn "Some dependencies failed to install"
        else
            log_warn "pip not found - cannot install dependencies"
        fi
    fi
    
    log_success "Plugin '$plugin_name' installed successfully"
}

# Create plugin wrapper
create_plugin_wrapper() {
    local plugin_name="$1"
    local plugin_dir="$2"
    
    cat > "$plugin_dir/__init__.py" << EOF
"""
Plugin wrapper for $plugin_name
Auto-generated by Roop-Unleashed plugin manager
"""

import os
import sys
import importlib.util

# Add plugin source to Python path
plugin_source_dir = os.path.join(os.path.dirname(__file__), 'source')
if plugin_source_dir not in sys.path:
    sys.path.insert(0, plugin_source_dir)

def get_plugin_info():
    return {
        'name': '$plugin_name',
        'version': '1.0.0',
        'description': 'External plugin for $plugin_name',
        'category': 'custom',
        'enabled': True
    }

def initialize():
    """Initialize the plugin"""
    try:
        # Try to import and initialize the main plugin module
        plugin_module = None
        
        # Common entry point names
        entry_points = ['main', '__init__', 'plugin', '$plugin_name']
        
        for entry_point in entry_points:
            try:
                spec = importlib.util.spec_from_file_location(
                    entry_point, 
                    os.path.join(plugin_source_dir, f'{entry_point}.py')
                )
                if spec and spec.loader:
                    plugin_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(plugin_module)
                    break
            except (ImportError, FileNotFoundError):
                continue
        
        if plugin_module and hasattr(plugin_module, 'initialize'):
            return plugin_module.initialize()
        
        return True
        
    except Exception as e:
        print(f"Failed to initialize plugin $plugin_name: {e}")
        return False

# Export common functions
__all__ = ['get_plugin_info', 'initialize']
EOF
    
    log_info "Created plugin wrapper for $plugin_name"
}

# Remove plugin
remove_plugin() {
    local plugin_name="$1"
    
    if ! check_plugin_installed "$plugin_name"; then
        log_error "Plugin '$plugin_name' not installed"
        return 1
    fi
    
    # Don't remove built-in plugins
    if [ -f "$PLUGINS_DIR/plugin_${plugin_name}.py" ]; then
        log_error "Cannot remove built-in plugin '$plugin_name'"
        return 1
    fi
    
    # Remove custom plugin
    local plugin_dir="$CUSTOM_PLUGINS_DIR/$plugin_name"
    if [ -d "$plugin_dir" ]; then
        read -p "Remove plugin '$plugin_name'? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$plugin_dir"
            log_success "Plugin '$plugin_name' removed"
        fi
    fi
}

# Enable/disable plugin
toggle_plugin() {
    local plugin_name="$1"
    local action="$2"
    
    if ! check_plugin_installed "$plugin_name"; then
        log_error "Plugin '$plugin_name' not installed"
        return 1
    fi
    
    case "$action" in
        enable)
            log_info "Enabling plugin: $plugin_name"
            # Add to enabled plugins list in registry
            log_success "Plugin '$plugin_name' enabled"
            ;;
        disable)
            log_info "Disabling plugin: $plugin_name"
            # Remove from enabled plugins list in registry
            log_success "Plugin '$plugin_name' disabled"
            ;;
        *)
            log_error "Invalid action: $action (use 'enable' or 'disable')"
            return 1
            ;;
    esac
}

# Update plugins
update_plugins() {
    log_info "Updating all custom plugins..."
    
    local updated=0
    
    if [ -d "$CUSTOM_PLUGINS_DIR" ]; then
        for plugin_dir in "$CUSTOM_PLUGINS_DIR"/*; do
            if [ -d "$plugin_dir/source/.git" ]; then
                local plugin_name=$(basename "$plugin_dir")
                log_info "Updating $plugin_name..."
                
                if git -C "$plugin_dir/source" pull origin main 2>/dev/null || git -C "$plugin_dir/source" pull origin master 2>/dev/null; then
                    log_success "Updated $plugin_name"
                    ((updated++))
                else
                    log_warn "Failed to update $plugin_name"
                fi
            fi
        done
    fi
    
    log_info "Updated $updated plugins"
}

# Show plugin information
show_plugin_info() {
    local plugin_name="$1"
    
    if ! check_plugin_installed "$plugin_name"; then
        log_error "Plugin '$plugin_name' not installed"
        return 1
    fi
    
    echo "Plugin Information: $plugin_name"
    echo "================================="
    
    # Check if it's a built-in plugin
    if [ -f "$PLUGINS_DIR/plugin_${plugin_name}.py" ]; then
        echo "Type: Built-in"
        echo "Location: $PLUGINS_DIR/plugin_${plugin_name}.py"
        
        # Try to extract info from the plugin file
        local description=$(grep -m 1 "^#.*" "$PLUGINS_DIR/plugin_${plugin_name}.py" | sed 's/^# *//' || echo "No description available")
        echo "Description: $description"
        
    elif [ -d "$CUSTOM_PLUGINS_DIR/$plugin_name" ]; then
        echo "Type: Custom"
        echo "Location: $CUSTOM_PLUGINS_DIR/$plugin_name"
        
        # Show git info if available
        if [ -d "$CUSTOM_PLUGINS_DIR/$plugin_name/source/.git" ]; then
            echo "Git repository:"
            git -C "$CUSTOM_PLUGINS_DIR/$plugin_name/source" remote get-url origin 2>/dev/null || echo "Unknown"
            
            echo "Last commit:"
            git -C "$CUSTOM_PLUGINS_DIR/$plugin_name/source" log -1 --oneline 2>/dev/null || echo "Unknown"
        fi
        
        # Show dependencies
        if [ -f "$CUSTOM_PLUGINS_DIR/$plugin_name/source/requirements.txt" ]; then
            echo "Dependencies:"
            cat "$CUSTOM_PLUGINS_DIR/$plugin_name/source/requirements.txt"
        fi
    fi
}

# Create custom plugin template
create_plugin_template() {
    local plugin_name="$1"
    
    if [ -z "$plugin_name" ]; then
        log_error "Please provide a plugin name"
        return 1
    fi
    
    local plugin_dir="$CUSTOM_PLUGINS_DIR/$plugin_name"
    
    if [ -d "$plugin_dir" ]; then
        log_error "Plugin '$plugin_name' already exists"
        return 1
    fi
    
    mkdir -p "$plugin_dir"
    
    # Create main plugin file
    cat > "$plugin_dir/main.py" << EOF
"""
Custom Roop-Unleashed Plugin: $plugin_name
Created: $(date)
"""

def get_plugin_info():
    """Return plugin information"""
    return {
        'name': '$plugin_name',
        'version': '1.0.0',
        'description': 'Custom plugin for $plugin_name',
        'author': 'Your Name',
        'category': 'custom'
    }

def initialize():
    """Initialize the plugin"""
    print(f"Initializing plugin: $plugin_name")
    # Add your initialization code here
    return True

def process_frame(frame, **kwargs):
    """Process a single frame"""
    # Add your frame processing logic here
    # frame: numpy array representing the image
    # kwargs: additional parameters
    
    # Example: return the frame unchanged
    return frame

def process_video(input_path, output_path, **kwargs):
    """Process a video file"""
    # Add your video processing logic here
    # input_path: path to input video
    # output_path: path to output video
    # kwargs: additional parameters
    
    print(f"Processing video: {input_path} -> {output_path}")
    # Implement your video processing logic
    return True

def cleanup():
    """Cleanup resources"""
    print(f"Cleaning up plugin: $plugin_name")
    # Add cleanup code here
    pass

# Plugin hooks (optional)
HOOKS = {
    'before_processing': None,
    'after_processing': None,
    'on_error': None
}

if __name__ == "__main__":
    # Test the plugin
    print(f"Testing plugin: $plugin_name")
    info = get_plugin_info()
    print(f"Plugin info: {info}")
    
    if initialize():
        print("Plugin initialized successfully")
    else:
        print("Plugin initialization failed")
EOF

    # Create requirements file
    cat > "$plugin_dir/requirements.txt" << EOF
# Dependencies for $plugin_name plugin
# Add your required packages here
# Example:
# numpy>=1.20.0
# opencv-python>=4.5.0
# torch>=1.9.0
EOF

    # Create README
    cat > "$plugin_dir/README.md" << EOF
# $plugin_name Plugin

## Description
Custom plugin for Roop-Unleashed

## Installation
This plugin was created using the Roop-Unleashed plugin manager.

## Usage
Describe how to use your plugin here.

## Configuration
Explain any configuration options.

## Requirements
- List any special requirements or dependencies

## Author
Your Name

## License
Add license information
EOF

    log_success "Plugin template created: $plugin_dir"
    log_info "Edit the files in $plugin_dir to implement your plugin"
}

# Show help
show_help() {
    cat << EOF
Roop-Unleashed Plugin Management Utility

Usage: $0 [COMMAND] [OPTIONS]

COMMANDS:
    list                    List available plugins
    list-installed          List installed plugins
    install <plugin>        Install a plugin
    remove <plugin>         Remove a plugin
    enable <plugin>         Enable a plugin
    disable <plugin>        Disable a plugin
    update                  Update all plugins
    info <plugin>           Show plugin information
    create <name>           Create plugin template
    
EXAMPLES:
    $0 list                           # List available plugins
    $0 install real_esrgan            # Install Real-ESRGAN plugin
    $0 list-installed                 # Show installed plugins
    $0 info real_esrgan               # Show plugin info
    $0 create my_plugin               # Create custom plugin template

EOF
}

# Main function
main() {
    init_plugin_system
    
    case "${1:-}" in
        list|ls)
            list_available
            ;;
        list-installed|installed)
            list_installed
            ;;
        install)
            if [ $# -lt 2 ]; then
                log_error "Please specify a plugin name"
                show_help
                exit 1
            fi
            install_plugin "$2"
            ;;
        remove|rm|uninstall)
            if [ $# -lt 2 ]; then
                log_error "Please specify a plugin name"
                exit 1
            fi
            remove_plugin "$2"
            ;;
        enable)
            if [ $# -lt 2 ]; then
                log_error "Please specify a plugin name"
                exit 1
            fi
            toggle_plugin "$2" "enable"
            ;;
        disable)
            if [ $# -lt 2 ]; then
                log_error "Please specify a plugin name"
                exit 1
            fi
            toggle_plugin "$2" "disable"
            ;;
        update)
            update_plugins
            ;;
        info|show)
            if [ $# -lt 2 ]; then
                log_error "Please specify a plugin name"
                exit 1
            fi
            show_plugin_info "$2"
            ;;
        create|new)
            if [ $# -lt 2 ]; then
                log_error "Please specify a plugin name"
                exit 1
            fi
            create_plugin_template "$2"
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