#!/usr/bin/env bash

# Roop-Unleashed Utility Manager
# Central hub for all utility scripts and tools

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Show main menu
show_main_menu() {
    clear
    echo -e "${BOLD}${CYAN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                  ROOP-UNLEASHED UTILITY MANAGER             ║
║                     Enhanced Linux Tools                     ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    echo ""
    echo -e "${BOLD}Available Tools:${NC}"
    echo ""
    echo -e "${GREEN}1)${NC} ${BOLD}Installation & Setup${NC}"
    echo "   - Enhanced Linux installer with GPU detection"
    echo "   - System requirements checking"
    echo "   - Backup and restore capabilities"
    echo ""
    echo -e "${GREEN}2)${NC} ${BOLD}Model Management${NC}"
    echo "   - Download and manage AI models"
    echo "   - Model verification and integrity checking"
    echo "   - Storage optimization"
    echo ""
    echo -e "${GREEN}3)${NC} ${BOLD}Plugin Management${NC}"
    echo "   - Install and manage plugins"
    echo "   - Plugin development tools"
    echo "   - Custom plugin creation"
    echo ""
    echo -e "${GREEN}4)${NC} ${BOLD}System Diagnostics${NC}"
    echo "   - Comprehensive system health check"
    echo "   - Performance benchmarking"
    echo "   - Troubleshooting tools"
    echo ""
    echo -e "${GREEN}5)${NC} ${BOLD}Batch Processing${NC}"
    echo "   - Bulk image/video processing"
    echo "   - Queue management"
    echo "   - Progress monitoring"
    echo ""
    echo -e "${GREEN}6)${NC} ${BOLD}Configuration & Optimization${NC}"
    echo "   - Performance tuning"
    echo "   - Memory optimization"
    echo "   - GPU configuration"
    echo ""
    echo -e "${GREEN}7)${NC} ${BOLD}Backup & Restore${NC}"
    echo "   - Model and configuration backup"
    echo "   - Environment snapshots"
    echo "   - Disaster recovery"
    echo ""
    echo -e "${GREEN}8)${NC} ${BOLD}Help & Documentation${NC}"
    echo "   - User guides and tutorials"
    echo "   - Troubleshooting guides"
    echo "   - FAQ and common issues"
    echo ""
    echo -e "${YELLOW}q)${NC} ${BOLD}Quit${NC}"
    echo ""
}

# Installation menu
installation_menu() {
    while true; do
        clear
        echo -e "${BOLD}${CYAN}Installation & Setup${NC}"
        echo "===================="
        echo ""
        echo "1) Run enhanced Linux installer"
        echo "2) Check system requirements"
        echo "3) Install with CPU-only mode"
        echo "4) Install with verbose logging"
        echo "5) Backup current installation"
        echo "6) Restore from backup"
        echo "7) Update installation"
        echo "b) Back to main menu"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1)
                echo ""
                log_info "Running enhanced Linux installer..."
                "$SCRIPT_DIR/install_linux_enhanced.sh"
                read -p "Press Enter to continue..."
                ;;
            2)
                echo ""
                log_info "Checking system requirements..."
                "$SCRIPT_DIR/system_diagnostics.sh" quick-check
                read -p "Press Enter to continue..."
                ;;
            3)
                echo ""
                log_info "Installing with CPU-only mode..."
                "$SCRIPT_DIR/install_linux_enhanced.sh" --force-cpu
                read -p "Press Enter to continue..."
                ;;
            4)
                echo ""
                log_info "Installing with verbose logging..."
                "$SCRIPT_DIR/install_linux_enhanced.sh" --verbose
                read -p "Press Enter to continue..."
                ;;
            5)
                echo ""
                log_info "Creating backup..."
                "$SCRIPT_DIR/install_linux_enhanced.sh" --backup-only
                read -p "Press Enter to continue..."
                ;;
            6)
                echo ""
                log_info "Restoring from backup..."
                "$SCRIPT_DIR/install_linux_enhanced.sh" --restore
                read -p "Press Enter to continue..."
                ;;
            7)
                echo ""
                log_info "Updating installation..."
                # Add update logic here
                log_warn "Update functionality coming soon"
                read -p "Press Enter to continue..."
                ;;
            b|back)
                break
                ;;
            *)
                log_error "Invalid option"
                sleep 1
                ;;
        esac
    done
}

# Model management menu
model_menu() {
    while true; do
        clear
        echo -e "${BOLD}${CYAN}Model Management${NC}"
        echo "================"
        echo ""
        echo "1) List available models"
        echo "2) Download required models"
        echo "3) Download all models"
        echo "4) Download specific model"
        echo "5) Verify model integrity"
        echo "6) Show storage usage"
        echo "7) Backup models"
        echo "8) Remove model"
        echo "b) Back to main menu"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1)
                echo ""
                "$SCRIPT_DIR/model_manager.sh" list
                read -p "Press Enter to continue..."
                ;;
            2)
                echo ""
                "$SCRIPT_DIR/model_manager.sh" download-required
                read -p "Press Enter to continue..."
                ;;
            3)
                echo ""
                "$SCRIPT_DIR/model_manager.sh" download-all
                read -p "Press Enter to continue..."
                ;;
            4)
                echo ""
                read -p "Enter model name: " model_name
                "$SCRIPT_DIR/model_manager.sh" download "$model_name"
                read -p "Press Enter to continue..."
                ;;
            5)
                echo ""
                "$SCRIPT_DIR/model_manager.sh" verify
                read -p "Press Enter to continue..."
                ;;
            6)
                echo ""
                "$SCRIPT_DIR/model_manager.sh" usage
                read -p "Press Enter to continue..."
                ;;
            7)
                echo ""
                "$SCRIPT_DIR/model_manager.sh" backup
                read -p "Press Enter to continue..."
                ;;
            8)
                echo ""
                read -p "Enter model name to remove: " model_name
                "$SCRIPT_DIR/model_manager.sh" remove "$model_name"
                read -p "Press Enter to continue..."
                ;;
            b|back)
                break
                ;;
            *)
                log_error "Invalid option"
                sleep 1
                ;;
        esac
    done
}

# Plugin management menu
plugin_menu() {
    while true; do
        clear
        echo -e "${BOLD}${CYAN}Plugin Management${NC}"
        echo "================="
        echo ""
        echo "1) List available plugins"
        echo "2) List installed plugins"
        echo "3) Install plugin"
        echo "4) Remove plugin"
        echo "5) Update plugins"
        echo "6) Plugin information"
        echo "7) Create custom plugin"
        echo "8) Enable/disable plugin"
        echo "b) Back to main menu"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1)
                echo ""
                "$SCRIPT_DIR/plugin_manager.sh" list
                read -p "Press Enter to continue..."
                ;;
            2)
                echo ""
                "$SCRIPT_DIR/plugin_manager.sh" list-installed
                read -p "Press Enter to continue..."
                ;;
            3)
                echo ""
                read -p "Enter plugin name: " plugin_name
                "$SCRIPT_DIR/plugin_manager.sh" install "$plugin_name"
                read -p "Press Enter to continue..."
                ;;
            4)
                echo ""
                read -p "Enter plugin name: " plugin_name
                "$SCRIPT_DIR/plugin_manager.sh" remove "$plugin_name"
                read -p "Press Enter to continue..."
                ;;
            5)
                echo ""
                "$SCRIPT_DIR/plugin_manager.sh" update
                read -p "Press Enter to continue..."
                ;;
            6)
                echo ""
                read -p "Enter plugin name: " plugin_name
                "$SCRIPT_DIR/plugin_manager.sh" info "$plugin_name"
                read -p "Press Enter to continue..."
                ;;
            7)
                echo ""
                read -p "Enter new plugin name: " plugin_name
                "$SCRIPT_DIR/plugin_manager.sh" create "$plugin_name"
                read -p "Press Enter to continue..."
                ;;
            8)
                echo ""
                read -p "Enter plugin name: " plugin_name
                read -p "Enable or disable? (e/d): " action
                case $action in
                    e|enable)
                        "$SCRIPT_DIR/plugin_manager.sh" enable "$plugin_name"
                        ;;
                    d|disable)
                        "$SCRIPT_DIR/plugin_manager.sh" disable "$plugin_name"
                        ;;
                    *)
                        log_error "Invalid action"
                        ;;
                esac
                read -p "Press Enter to continue..."
                ;;
            b|back)
                break
                ;;
            *)
                log_error "Invalid option"
                sleep 1
                ;;
        esac
    done
}

# System diagnostics menu
diagnostics_menu() {
    while true; do
        clear
        echo -e "${BOLD}${CYAN}System Diagnostics${NC}"
        echo "=================="
        echo ""
        echo "1) Quick health check"
        echo "2) Full system report"
        echo "3) GPU diagnostics"
        echo "4) Python environment check"
        echo "5) Performance benchmark"
        echo "6) Network connectivity test"
        echo "7) Fix common issues"
        echo "8) View latest report"
        echo "b) Back to main menu"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1)
                echo ""
                "$SCRIPT_DIR/system_diagnostics.sh" quick-check
                read -p "Press Enter to continue..."
                ;;
            2)
                echo ""
                "$SCRIPT_DIR/system_diagnostics.sh" full-report
                read -p "Press Enter to continue..."
                ;;
            3)
                echo ""
                "$SCRIPT_DIR/system_diagnostics.sh" gpu-check
                read -p "Press Enter to continue..."
                ;;
            4)
                echo ""
                "$SCRIPT_DIR/system_diagnostics.sh" python-check
                read -p "Press Enter to continue..."
                ;;
            5)
                echo ""
                "$SCRIPT_DIR/system_diagnostics.sh" performance-test
                read -p "Press Enter to continue..."
                ;;
            6)
                echo ""
                "$SCRIPT_DIR/system_diagnostics.sh" network-check
                read -p "Press Enter to continue..."
                ;;
            7)
                echo ""
                "$SCRIPT_DIR/system_diagnostics.sh" fix-common
                read -p "Press Enter to continue..."
                ;;
            8)
                echo ""
                latest_report=$(ls -t /tmp/roop_diagnostics_*.log 2>/dev/null | head -1)
                if [ -n "$latest_report" ]; then
                    echo "Latest report: $latest_report"
                    echo ""
                    tail -50 "$latest_report"
                else
                    log_warn "No diagnostic reports found"
                fi
                read -p "Press Enter to continue..."
                ;;
            b|back)
                break
                ;;
            *)
                log_error "Invalid option"
                sleep 1
                ;;
        esac
    done
}

# Batch processing menu
batch_menu() {
    while true; do
        clear
        echo -e "${BOLD}${CYAN}Batch Processing${NC}"
        echo "================"
        echo ""
        echo "1) Batch image processing"
        echo "2) Batch video processing"
        echo "3) Queue status"
        echo "4) Process queue"
        echo "5) Clear queue"
        echo "b) Back to main menu"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1|2|3|4|5)
                echo ""
                log_warn "Batch processing functionality coming soon"
                read -p "Press Enter to continue..."
                ;;
            b|back)
                break
                ;;
            *)
                log_error "Invalid option"
                sleep 1
                ;;
        esac
    done
}

# Configuration menu
config_menu() {
    while true; do
        clear
        echo -e "${BOLD}${CYAN}Configuration & Optimization${NC}"
        echo "============================="
        echo ""
        echo "1) GPU configuration"
        echo "2) Memory optimization"
        echo "3) Performance tuning"
        echo "4) Default settings"
        echo "5) Export configuration"
        echo "6) Import configuration"
        echo "b) Back to main menu"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1|2|3|4|5|6)
                echo ""
                log_warn "Configuration functionality coming soon"
                read -p "Press Enter to continue..."
                ;;
            b|back)
                break
                ;;
            *)
                log_error "Invalid option"
                sleep 1
                ;;
        esac
    done
}

# Backup menu
backup_menu() {
    while true; do
        clear
        echo -e "${BOLD}${CYAN}Backup & Restore${NC}"
        echo "================"
        echo ""
        echo "1) Create full backup"
        echo "2) Create models backup"
        echo "3) Create settings backup"
        echo "4) List backups"
        echo "5) Restore from backup"
        echo "6) Delete old backups"
        echo "b) Back to main menu"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1|2|3|4|5|6)
                echo ""
                log_warn "Advanced backup functionality coming soon"
                read -p "Press Enter to continue..."
                ;;
            b|back)
                break
                ;;
            *)
                log_error "Invalid option"
                sleep 1
                ;;
        esac
    done
}

# Help menu
help_menu() {
    while true; do
        clear
        echo -e "${BOLD}${CYAN}Help & Documentation${NC}"
        echo "===================="
        echo ""
        echo "1) User guide"
        echo "2) Installation guide"
        echo "3) Troubleshooting guide"
        echo "4) FAQ"
        echo "5) System requirements"
        echo "6) Performance tips"
        echo "7) Plugin development guide"
        echo "8) About"
        echo "b) Back to main menu"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1)
                show_user_guide
                ;;
            2)
                show_installation_guide
                ;;
            3)
                show_troubleshooting_guide
                ;;
            4)
                show_faq
                ;;
            5)
                show_system_requirements
                ;;
            6)
                show_performance_tips
                ;;
            7)
                show_plugin_dev_guide
                ;;
            8)
                show_about
                ;;
            b|back)
                break
                ;;
            *)
                log_error "Invalid option"
                sleep 1
                ;;
        esac
    done
}

# Help content functions
show_user_guide() {
    clear
    echo -e "${BOLD}${CYAN}User Guide${NC}"
    echo "=========="
    echo ""
    cat << 'EOF'
Roop-Unleashed is an advanced deepfake tool with the following features:

1. INSTALLATION:
   - Use the enhanced installer: ./scripts/install_linux_enhanced.sh
   - The installer automatically detects your system and GPU
   - Follow the prompts for a smooth installation

2. MODELS:
   - Download required models using the model manager
   - Models are automatically organized by type
   - Verify model integrity regularly

3. PLUGINS:
   - Extend functionality with additional plugins
   - Install plugins from the registry or create custom ones
   - Enable/disable plugins as needed

4. USAGE:
   - Launch with: ./start_roop_unleashed.sh
   - Use the web interface for easy operation
   - Monitor system resources during processing

5. TROUBLESHOOTING:
   - Run system diagnostics for issues
   - Check logs for error messages
   - Use the fix-common-issues tool

EOF
    read -p "Press Enter to continue..."
}

show_installation_guide() {
    clear
    echo -e "${BOLD}${CYAN}Installation Guide${NC}"
    echo "=================="
    echo ""
    cat << 'EOF'
SYSTEM REQUIREMENTS:
- Linux x86_64 system
- 8GB RAM (16GB recommended)
- 10GB free disk space
- Python 3.9-3.12
- CUDA-compatible GPU (optional but recommended)

INSTALLATION STEPS:

1. Clone the repository:
   git clone https://github.com/cbwinslow/roop-unleashed
   cd roop-unleashed

2. Run the enhanced installer:
   ./scripts/install_linux_enhanced.sh

3. The installer will:
   - Check system requirements
   - Install system dependencies
   - Detect and configure GPU
   - Create conda environment
   - Install Python packages
   - Download required models

4. Launch the application:
   ./start_roop_unleashed.sh

MANUAL INSTALLATION:
If the automatic installer fails, you can install manually:
- Install Miniconda
- Create Python 3.10 environment
- Install requirements.txt
- Download models manually

EOF
    read -p "Press Enter to continue..."
}

show_troubleshooting_guide() {
    clear
    echo -e "${BOLD}${CYAN}Troubleshooting Guide${NC}"
    echo "===================="
    echo ""
    cat << 'EOF'
COMMON ISSUES:

1. INSTALLATION FAILS:
   - Check system requirements
   - Ensure stable internet connection
   - Run with --verbose for detailed logs
   - Check available disk space

2. GPU NOT DETECTED:
   - Install NVIDIA drivers
   - Check CUDA installation
   - Restart system after driver installation
   - Use --force-cpu for CPU-only mode

3. PYTHON IMPORT ERRORS:
   - Activate the correct environment
   - Reinstall requirements.txt
   - Check Python version compatibility
   - Clear Python cache files

4. MODEL DOWNLOAD FAILS:
   - Check internet connectivity
   - Verify available disk space
   - Use model manager to retry download
   - Download models manually if needed

5. PERFORMANCE ISSUES:
   - Check GPU utilization
   - Monitor system resources
   - Optimize settings for your hardware
   - Close other applications

DIAGNOSTIC COMMANDS:
- ./scripts/utility_manager.sh (option 4)
- ./scripts/system_diagnostics.sh quick-check
- ./scripts/system_diagnostics.sh full-report

EOF
    read -p "Press Enter to continue..."
}

show_faq() {
    clear
    echo -e "${BOLD}${CYAN}Frequently Asked Questions${NC}"
    echo "=========================="
    echo ""
    cat << 'EOF'
Q: What are the minimum system requirements?
A: Linux x86_64, 8GB RAM, Python 3.9+, 10GB disk space

Q: Do I need a GPU?
A: No, but GPU significantly improves performance. Supports NVIDIA, AMD, and Intel.

Q: How do I update the application?
A: Use git pull and reinstall requirements, or use the update function.

Q: Can I use custom models?
A: Yes, place them in the models directory and configure in settings.

Q: How do I add new plugins?
A: Use the plugin manager or create custom plugins with templates.

Q: Why is processing slow?
A: Check GPU utilization, system resources, and model settings.

Q: How do I backup my installation?
A: Use the backup functions in the utility manager.

Q: Where are the logs stored?
A: Check /tmp for diagnostic logs and installer logs.

Q: Can I run this on a server?
A: Yes, but ensure proper GPU drivers and dependencies.

Q: How do I report bugs?
A: Create an issue on GitHub with system info and logs.

EOF
    read -p "Press Enter to continue..."
}

show_system_requirements() {
    clear
    echo -e "${BOLD}${CYAN}System Requirements${NC}"
    echo "=================="
    echo ""
    cat << 'EOF'
MINIMUM REQUIREMENTS:
- Operating System: Linux x86_64
- Memory: 8GB RAM
- Storage: 10GB free space
- Python: 3.9 - 3.12
- Internet: Required for installation and model downloads

RECOMMENDED REQUIREMENTS:
- Operating System: Ubuntu 20.04+ or equivalent
- Memory: 16GB+ RAM
- Storage: 50GB+ SSD storage
- GPU: NVIDIA GPU with 6GB+ VRAM
- Python: 3.10
- Internet: High-speed connection for model downloads

GPU SUPPORT:
- NVIDIA: CUDA 11.8 or 12.1
- AMD: ROCm 5.0+
- Intel: Intel GPU drivers
- CPU: Supported but slower

SUPPORTED DISTRIBUTIONS:
- Ubuntu/Debian and derivatives
- Fedora/RHEL/CentOS
- Arch Linux/Manjaro
- openSUSE
- Others with manual dependency installation

EOF
    read -p "Press Enter to continue..."
}

show_performance_tips() {
    clear
    echo -e "${BOLD}${CYAN}Performance Tips${NC}"
    echo "==============="
    echo ""
    cat << 'EOF'
HARDWARE OPTIMIZATION:
- Use NVIDIA GPU with CUDA for best performance
- Ensure adequate GPU memory (6GB+ recommended)
- Use SSD storage for faster model loading
- Close unnecessary applications

SOFTWARE OPTIMIZATION:
- Keep GPU drivers updated
- Use appropriate batch sizes
- Enable GPU acceleration in settings
- Monitor system resources during processing

MODEL OPTIMIZATION:
- Use appropriate model resolution for your needs
- Consider quality vs speed tradeoffs
- Download optimized model variants
- Keep models on fast storage

SYSTEM OPTIMIZATION:
- Increase swap space if needed
- Set CPU governor to performance
- Disable unnecessary services
- Use dedicated processing sessions

TROUBLESHOOTING PERFORMANCE:
- Check GPU utilization with nvidia-smi
- Monitor memory usage
- Profile processing times
- Test with different settings

EOF
    read -p "Press Enter to continue..."
}

show_plugin_dev_guide() {
    clear
    echo -e "${BOLD}${CYAN}Plugin Development Guide${NC}"
    echo "======================="
    echo ""
    cat << 'EOF'
CREATING PLUGINS:

1. Use the plugin manager to create a template:
   ./scripts/plugin_manager.sh create my_plugin

2. Edit the generated files:
   - main.py: Core plugin logic
   - requirements.txt: Dependencies
   - README.md: Documentation

3. Implement required functions:
   - get_plugin_info(): Plugin metadata
   - initialize(): Setup code
   - process_frame(): Frame processing
   - process_video(): Video processing
   - cleanup(): Resource cleanup

4. Test your plugin:
   python plugins/custom/my_plugin/main.py

5. Install and enable:
   ./scripts/plugin_manager.sh enable my_plugin

PLUGIN HOOKS:
- before_processing: Called before processing starts
- after_processing: Called after processing completes
- on_error: Called when errors occur

BEST PRACTICES:
- Follow Python coding standards
- Include comprehensive error handling
- Document all functions and parameters
- Test with various input types
- Include example usage

EOF
    read -p "Press Enter to continue..."
}

show_about() {
    clear
    echo -e "${BOLD}${CYAN}About Roop-Unleashed${NC}"
    echo "==================="
    echo ""
    cat << 'EOF'
Roop-Unleashed Enhanced Linux Tools
Version: 1.0.0
Author: Enhanced by AI Assistant

FEATURES:
- Robust Linux installation system
- Comprehensive model management
- Advanced plugin system
- System diagnostics and troubleshooting
- Performance optimization tools
- Backup and restore capabilities

CREDITS:
- Original Roop project
- InsightFace team
- PyTorch and OpenCV communities
- All contributing developers

LICENSE:
- Check repository for license information
- Respect all model and dependency licenses
- Use responsibly and ethically

SUPPORT:
- GitHub Issues for bug reports
- Documentation in repository
- Community forums and discussions

DISCLAIMER:
This software is for educational and research purposes.
Users are responsible for ethical use and legal compliance.

EOF
    read -p "Press Enter to continue..."
}

# Main application loop
main() {
    # Check if we're in the right directory
    if [ ! -f "$REPO_DIR/run.py" ]; then
        log_error "Not in Roop-Unleashed repository directory"
        log_error "Please run this script from the repository root"
        exit 1
    fi
    
    # Make sure all scripts are executable
    chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true
    
    while true; do
        show_main_menu
        
        read -p "Select option (1-8, q): " choice
        
        case $choice in
            1)
                installation_menu
                ;;
            2)
                model_menu
                ;;
            3)
                plugin_menu
                ;;
            4)
                diagnostics_menu
                ;;
            5)
                batch_menu
                ;;
            6)
                config_menu
                ;;
            7)
                backup_menu
                ;;
            8)
                help_menu
                ;;
            q|quit|exit)
                echo ""
                log_info "Thank you for using Roop-Unleashed Enhanced Tools!"
                exit 0
                ;;
            *)
                log_error "Invalid option"
                sleep 1
                ;;
        esac
    done
}

main "$@"