# Roop-Unleashed Enhanced Linux Scripts

This directory contains comprehensive utility scripts for managing and optimizing your Roop-Unleashed installation on Linux systems.

## Scripts Overview

### 🚀 Main Utility Manager
- **`utility_manager.sh`** - Central hub for all tools with interactive menu system

### 📦 Installation & Setup
- **`install_linux_enhanced.sh`** - Robust one-click installer with comprehensive error handling
  - Multi-distribution support (Ubuntu, Fedora, Arch, openSUSE, etc.)
  - Automatic GPU detection and driver setup
  - Backup and restore capabilities
  - System requirements validation

### 🤖 Model Management
- **`model_manager.sh`** - Download, verify, and manage AI models
  - Curated model registry with latest models
  - Automatic model organization
  - Integrity verification
  - Storage optimization

### 🔌 Plugin Management
- **`plugin_manager.sh`** - Extend functionality with plugins
  - Plugin discovery and installation
  - Custom plugin development tools
  - Plugin dependency management
  - Plugin templates for developers

### 🔧 System Diagnostics
- **`system_diagnostics.sh`** - Comprehensive system analysis
  - Hardware compatibility checking
  - Performance benchmarking
  - GPU diagnostics
  - Network connectivity testing
  - Automated issue resolution

## Quick Start

### Easy Interactive Mode
```bash
# Launch the main utility manager (recommended for beginners)
./scripts/utility_manager.sh
```

### Direct Installation
```bash
# Run the enhanced installer directly
./scripts/install_linux_enhanced.sh

# CPU-only installation
./scripts/install_linux_enhanced.sh --force-cpu

# Verbose installation with detailed logging
./scripts/install_linux_enhanced.sh --verbose
```

### Model Management
```bash
# List all available models
./scripts/model_manager.sh list

# Download required models
./scripts/model_manager.sh download-required

# Download specific model
./scripts/model_manager.sh download inswapper_128.onnx

# Verify model integrity
./scripts/model_manager.sh verify
```

### System Health Check
```bash
# Quick health check
./scripts/system_diagnostics.sh quick-check

# Comprehensive system report
./scripts/system_diagnostics.sh full-report

# GPU-specific diagnostics
./scripts/system_diagnostics.sh gpu-check
```

### Plugin Management
```bash
# List available plugins
./scripts/plugin_manager.sh list

# Install a plugin
./scripts/plugin_manager.sh install real_esrgan

# Create custom plugin template
./scripts/plugin_manager.sh create my_custom_plugin
```

## Features

### ✅ Enhanced Installation System
- **Multi-Distribution Support**: Works on Ubuntu, Fedora, Arch, openSUSE, and more
- **Automatic GPU Detection**: Supports NVIDIA (CUDA), AMD (ROCm), and Intel GPUs
- **Smart Dependency Management**: Automatically installs system and Python dependencies
- **Backup & Restore**: Automatic backup before installation with restore capabilities
- **Robust Error Handling**: Comprehensive error detection and recovery mechanisms
- **Progress Monitoring**: Detailed logging and progress reporting

### 🤖 Advanced Model Management
- **Curated Model Registry**: Pre-configured registry of the latest AI models
- **Automatic Organization**: Models sorted by type (face swapping, enhancement, detection)
- **Integrity Verification**: SHA checksums and file validation
- **Storage Optimization**: Efficient storage and duplicate detection
- **Model Discovery**: Find and evaluate new models from various sources

### 🔌 Comprehensive Plugin System
- **Plugin Registry**: Discover plugins for face enhancement, video processing, and utilities
- **Easy Installation**: One-command plugin installation with dependency management
- **Development Tools**: Templates and tools for creating custom plugins
- **Plugin Lifecycle**: Enable, disable, update, and remove plugins seamlessly

### 🔧 System Diagnostics & Optimization
- **Hardware Analysis**: Detailed system information and compatibility checking
- **Performance Benchmarking**: CPU and GPU performance testing
- **Issue Detection**: Automatic detection of common problems
- **Fix Recommendations**: Suggested solutions for detected issues
- **Resource Monitoring**: Memory, disk, and GPU utilization tracking

## System Requirements

### Minimum Requirements
- **OS**: Linux x86_64 (Ubuntu 18.04+, Fedora 30+, Arch, openSUSE Leap 15+)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space (50GB recommended)
- **Python**: 3.9 - 3.12
- **Internet**: Required for downloads

### GPU Support
- **NVIDIA**: CUDA 11.8 or 12.1+ (recommended)
- **AMD**: ROCm 5.0+
- **Intel**: Intel GPU drivers
- **CPU**: Supported but significantly slower

### Recommended Setup
- **GPU**: NVIDIA RTX series with 8GB+ VRAM
- **Storage**: NVMe SSD for model storage
- **RAM**: 32GB for large model processing
- **Network**: High-speed internet for model downloads

## Advanced Usage

### Installation Options
```bash
# Standard installation
./scripts/install_linux_enhanced.sh

# Force CPU-only mode
./scripts/install_linux_enhanced.sh --force-cpu

# Skip GPU detection and driver installation
./scripts/install_linux_enhanced.sh --skip-gpu-detect

# Verbose logging
./scripts/install_linux_enhanced.sh --verbose

# Backup existing installation only
./scripts/install_linux_enhanced.sh --backup-only

# Restore from latest backup
./scripts/install_linux_enhanced.sh --restore
```

### Model Management Commands
```bash
# Download all available models
./scripts/model_manager.sh download-all

# Show storage usage
./scripts/model_manager.sh usage

# Create backup of models
./scripts/model_manager.sh backup

# Remove specific model
./scripts/model_manager.sh remove model_name
```

### Diagnostic Commands
```bash
# Performance benchmark
./scripts/system_diagnostics.sh performance-test

# Network connectivity test
./scripts/system_diagnostics.sh network-check

# Attempt to fix common issues
./scripts/system_diagnostics.sh fix-common

# Python environment check
./scripts/system_diagnostics.sh python-check
```

## Troubleshooting

### Common Issues

#### Installation Fails
1. Check system requirements
2. Ensure stable internet connection
3. Run with `--verbose` flag for detailed logs
4. Check available disk space
5. Verify system dependencies

#### GPU Not Detected
1. Install latest GPU drivers
2. Restart system after driver installation
3. Check CUDA/ROCm installation
4. Use `--force-cpu` for CPU-only mode
5. Run GPU diagnostics

#### Python Import Errors
1. Activate correct environment
2. Reinstall requirements
3. Check Python version compatibility
4. Clear Python cache

#### Model Download Fails
1. Check internet connectivity
2. Verify available disk space
3. Retry with model manager
4. Download models manually

### Getting Help
1. Run system diagnostics: `./scripts/system_diagnostics.sh full-report`
2. Check log files in `/tmp/` directory
3. Use interactive utility manager for guided troubleshooting
4. Create GitHub issue with system information

## Development

### Contributing
1. Follow existing code structure and conventions
2. Add comprehensive error handling
3. Include usage examples and documentation
4. Test on multiple Linux distributions

### Creating Custom Plugins
1. Use plugin template: `./scripts/plugin_manager.sh create my_plugin`
2. Implement required functions in `main.py`
3. Add dependencies to `requirements.txt`
4. Test thoroughly before distribution

### Script Structure
- All scripts use bash with `set -euo pipefail` for robust error handling
- Consistent logging and color-coded output
- Modular design for easy maintenance
- Comprehensive help and usage information

## Security Considerations

- Scripts require sudo for system package installation
- All downloads are verified when possible
- No sensitive information is stored in logs
- User consent required for destructive operations

## License

These scripts are part of the Roop-Unleashed project. Please respect all software licenses and use responsibly.

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Run system diagnostics
3. Search existing GitHub issues
4. Create new issue with detailed information

---

**Note**: These tools are designed to make Roop-Unleashed installation and management as smooth as possible on Linux systems. Always backup important data before making system changes.