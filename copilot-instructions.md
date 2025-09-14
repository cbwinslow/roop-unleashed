# Copilot Instructions for Roop-Unleashed

## Project Overview

Roop-Unleashed is an advanced face-swapping application that provides uncensored deepfake capabilities for images and videos. This project focuses on high-quality face transfer, real-time processing, and comprehensive AI-driven optimization.

## Core Technologies

- **Python 3.9-3.12**: Main programming language
- **PyTorch 2.0-2.2**: Deep learning framework for face processing
- **ONNX Runtime**: Optimized inference engine
- **Gradio**: Web-based user interface
- **OpenCV**: Computer vision processing
- **InsightFace**: Face recognition and analysis
- **CUDA/ROCm/DirectML**: GPU acceleration support

## Architecture Principles

### Face Processing Pipeline
1. **Enhanced Face Detection**: Multi-scale detection with quality assessment
2. **Advanced Face Mapping**: Handles angles, obstructions, and facial features
3. **Intelligent Blending**: Multiple blend modes (Poisson, Multi-band, Gradient)
4. **Quality Assessment**: Real-time quality scoring and validation
5. **Performance Optimization**: GPU-accelerated processing with fallbacks

### AI Agent Framework
- **Multi-Agent System**: Coordinated agents for different tasks
- **RAG Integration**: Retrieval-augmented generation for intelligent assistance
- **Natural Language Processing**: Command understanding and execution
- **MCP Server**: Model Context Protocol for agent communication
- **Self-Healing**: Automatic error detection and recovery

## Key Modules and Their Purposes

### Core Modules
- `roop/core.py`: Main processing engine and coordination
- `roop/enhanced_face_detection.py`: Advanced face detection with quality metrics
- `roop/enhanced_face_swapper.py`: Improved face swapping algorithms
- `roop/advanced_blending.py`: Multiple blending techniques for natural results
- `roop/nvidia_optimizer.py`: Hardware-specific optimizations

### Agent System
- `agents/manager.py`: Central coordination of all AI agents
- `agents/enhanced_agents.py`: Advanced AI capabilities for processing
- `agents/model_agent.py`: Model management and optimization
- `agents/operation_agent.py`: Task execution and monitoring
- `agents/nlp_agent.py`: Natural language understanding

### Quality and Testing
- `tests/`: Comprehensive test suite covering all functionality
- `test_structure.py`: Structural validation and enhancement verification
- Performance benchmarking and metrics collection

## Coding Standards

### Code Style
- **Line Length**: Maximum 120 characters
- **Formatting**: Follow PEP 8 with project-specific extensions
- **Type Hints**: Use comprehensive type annotations
- **Documentation**: Docstrings for all public methods and classes
- **Error Handling**: Comprehensive exception handling with logging

### Testing Requirements
- **Unit Tests**: Cover individual functions and methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark processing speed and accuracy
- **Quality Tests**: Validate face-swapping results
- **Hardware Tests**: Verify GPU/CPU fallback behavior

### Performance Considerations
- **Memory Efficiency**: Optimize for large image/video processing
- **GPU Utilization**: Maximize parallel processing capabilities
- **Caching**: Implement intelligent caching for repeated operations
- **Streaming**: Support real-time video processing
- **Batch Processing**: Efficient handling of multiple files

## Face Processing Expertise

### Advanced Face Mapping
- **Angle Handling**: Robust detection and mapping at various face angles
- **Obstruction Management**: Process faces with glasses, masks, accessories
- **Feature Preservation**: Maintain scars, blemishes, makeup, and unique features
- **Hair Integration**: Include hair regions for enhanced realism
- **Lighting Adaptation**: Adjust for different lighting conditions

### Quality Metrics
- **Structural Similarity**: SSIM and PSNR measurements
- **Perceptual Quality**: LPIPS and other perceptual metrics
- **Face Identity**: Verification of successful identity transfer
- **Natural Appearance**: Assessment of blending quality
- **Temporal Consistency**: Frame-to-frame coherence in videos

### Optimization Targets
- **Speed**: Processing time per frame/image
- **Quality**: Visual fidelity and naturalness
- **Memory**: RAM and VRAM usage efficiency
- **Accuracy**: Face detection and alignment precision
- **Stability**: Consistent results across different inputs

## AI Agent Capabilities

### Monitoring and Analysis
- **Real-time Metrics**: Continuous performance monitoring
- **Error Detection**: Automatic identification of processing issues
- **Quality Assessment**: Ongoing evaluation of output quality
- **Resource Monitoring**: Track GPU/CPU/memory usage
- **Log Analysis**: Intelligent parsing of application logs

### Autonomous Optimization
- **Parameter Tuning**: Automatic adjustment of processing parameters
- **Model Selection**: Choose optimal models for specific scenarios
- **Resource Allocation**: Dynamic resource management
- **Error Recovery**: Automatic retry and fallback mechanisms
- **Performance Scaling**: Adaptive processing based on hardware

### Data Collection and Learning
- **Telemetry Gathering**: Comprehensive system metrics
- **User Behavior**: Processing patterns and preferences
- **Performance Benchmarks**: Continuous performance tracking
- **Error Patterns**: Analysis of common failure modes
- **Optimization Opportunities**: Identification of improvement areas

## Development Workflow

### Before Making Changes
1. **Understand Context**: Review related code and documentation
2. **Run Tests**: Ensure current functionality works
3. **Check Performance**: Baseline metrics before changes
4. **Plan Implementation**: Design with minimal disruption

### Implementation Guidelines
1. **Incremental Changes**: Small, testable modifications
2. **Backward Compatibility**: Maintain existing API contracts
3. **Error Handling**: Comprehensive exception management
4. **Logging**: Detailed logging for debugging and monitoring
5. **Documentation**: Update docs and comments

### Testing and Validation
1. **Unit Tests**: Test individual components
2. **Integration Tests**: Verify component interactions
3. **Performance Tests**: Ensure no regression in speed/quality
4. **Visual Validation**: Manual review of processing results
5. **CI/CD Verification**: All automated checks pass

## Specialized Knowledge Areas

### Computer Vision
- Deep understanding of face detection algorithms
- Knowledge of image processing techniques
- Familiarity with neural network architectures for vision
- Experience with ONNX model optimization
- Understanding of different color spaces and transformations

### GPU Programming
- CUDA programming for NVIDIA hardware
- ROCm for AMD GPU acceleration
- Memory management and optimization
- Kernel optimization and profiling
- Multi-GPU coordination

### Video Processing
- FFmpeg integration and optimization
- Frame-by-frame consistency
- Temporal coherence in face swapping
- Real-time streaming capabilities
- Codec optimization for different formats

### AI/ML Operations
- Model lifecycle management
- Performance monitoring and alerting
- Automated testing and validation
- Resource optimization and scaling
- Continuous integration for ML workflows

## Common Tasks and Patterns

### Adding New Face Processing Features
1. Implement in appropriate module (`enhanced_face_detection.py`, etc.)
2. Add comprehensive error handling
3. Include performance metrics
4. Create corresponding tests
5. Update configuration options
6. Document new capabilities

### Optimizing Performance
1. Profile current performance
2. Identify bottlenecks
3. Implement optimizations with A/B testing
4. Measure improvements
5. Update benchmarks
6. Document optimization techniques

### Enhancing AI Agents
1. Extend agent capabilities in `agents/` modules
2. Add new monitoring metrics
3. Implement decision-making logic
4. Create agent coordination protocols
5. Test autonomous behavior
6. Monitor agent performance

## Quality Assurance

### Testing Strategy
- **Automated Testing**: Comprehensive test suite with CI/CD
- **Performance Benchmarking**: Regular performance regression testing
- **Visual Quality Assessment**: Automated and manual quality checks
- **Hardware Compatibility**: Testing across different GPU/CPU configurations
- **Edge Case Handling**: Testing with challenging inputs

### Monitoring and Alerting
- **Real-time Metrics**: Live dashboard of system performance
- **Error Tracking**: Comprehensive error logging and analysis
- **Performance Alerts**: Automatic notification of performance degradation
- **Quality Metrics**: Continuous monitoring of output quality
- **Resource Usage**: Tracking of system resource consumption

## Future Development Directions

### Research Areas
- **Novel Face Mapping Techniques**: Exploring new approaches to face transfer
- **Real-time Quality Enhancement**: Improving processing speed without quality loss
- **Advanced Obstruction Handling**: Better processing of occluded faces
- **Hair and Background Integration**: More realistic full-head processing
- **Lighting and Color Adaptation**: Improved matching of target environments

### Technology Integration
- **Latest AI Models**: Integration of cutting-edge face processing models
- **Hardware Acceleration**: Support for new GPU architectures
- **Cloud Processing**: Distributed processing capabilities
- **Mobile Optimization**: Efficient processing on mobile devices
- **Web Assembly**: Browser-based processing capabilities

This document serves as a comprehensive guide for AI assistants working on the Roop-Unleashed project, ensuring consistent high-quality contributions and maintaining the project's technical excellence.