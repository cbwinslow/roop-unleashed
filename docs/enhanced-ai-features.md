# Enhanced AI Features for Roop Unleashed

This document describes the new enhanced AI features added to Roop Unleashed, including local LLM integration, RAG knowledge base, advanced agents, and NVIDIA optimizations.

## Overview

The enhanced version includes:

- **Local LLM Integration**: Support for Ollama and OpenAI-compatible local APIs
- **RAG System**: Knowledge base with vector search for intelligent assistance
- **Enhanced Agents**: Specialized AI agents for different tasks
- **NVIDIA Optimizations**: TensorRT, CUDA optimizations, and performance monitoring
- **Robust Error Handling**: Comprehensive error recovery and logging
- **Enhanced Configuration**: Expandable YAML configuration with validation

## Installation

### Basic Installation

The enhanced features are optional and gracefully degrade if dependencies aren't available:

```bash
# Basic installation (existing functionality)
pip install -r requirements.txt

# Enhanced AI features (optional)
pip install -r requirements-ai.txt
```

### Local LLM Setup (Ollama)

1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama3.2:3b`
3. Enable in configuration:

```yaml
ai_config:
  local_llm:
    enabled: true
    provider: "ollama"
    base_url: "http://localhost:11434"
    model: "llama3.2:3b"
```

## Configuration

### Enhanced Configuration Structure

The new configuration supports nested settings:

```yaml
# Basic Settings (existing)
selected_theme: "Default"
provider: "cuda"
max_threads: 4

# AI Configuration (new)
ai_config:
  local_llm:
    enabled: false
    provider: "ollama"  # or "local_api"
    base_url: "http://localhost:11434"
    model: "llama3.2:3b"
  
  rag:
    enabled: false
    embedding_model: "all-MiniLM-L6-v2"
    vector_store_path: "./rag_vectors"

# Agent Configuration
agents:
  rag_agent:
    enabled: false
    knowledge_base_path: "./knowledge"
  
  optimization_agent:
    enabled: true
    auto_optimize: false

# NVIDIA Optimizations
nvidia_config:
  tensorrt:
    enabled: false
    precision: "fp16"
  cuda:
    memory_fraction: 0.9
    streams: 2
```

### Configuration Migration

Existing configurations are automatically migrated:

```bash
# Check migration status
python -c "from roop.config_migrator import ConfigMigrator; print(ConfigMigrator('config.yaml', 'config.yaml', 'config_defaults.yaml').get_migration_report())"

# Perform migration
python -c "from roop.config_migrator import migrate_config; migrate_config()"
```

## Enhanced Agent System

### Available Agents

- **rag**: Knowledge base search and Q&A
- **video**: Video processing assistance
- **optimization**: Performance optimization advice
- **image_generation**: AI image generation help
- **troubleshooting**: Error diagnosis and resolution
- **installer**: Installation and dependency management (existing)
- **model**: Model management (existing)
- **operation**: General operations (existing)
- **nlp**: Natural language processing (existing)

### Usage Examples

```python
from agents.manager import MultiAgentManager

# Initialize with enhanced features
manager = MultiAgentManager(settings)

# Direct agent assistance
response = manager.assist('optimization', 'How to make processing faster?')

# Smart routing (auto-selects best agent)
response = manager.smart_assist('My video processing is slow')

# System status
status = manager.get_system_status()
```

### Command Line Interface

```bash
# Interactive help
python -m roop.help

# Direct agent queries (when running)
assist('video', 'How to optimize video encoding?')
smart_assist('GPU memory errors')
```

## RAG Knowledge Base

### Default Knowledge Base

The system includes built-in knowledge about:

- Face swapping best practices
- GPU optimization techniques
- Troubleshooting common issues
- Performance tuning

### Adding Custom Knowledge

```python
# Add knowledge through agent manager
manager.add_knowledge("""
Custom knowledge about face swapping:
- Use high-resolution source images
- Ensure good lighting conditions
""", source="user_guide")

# Or directly through RAG system
rag_system.add_knowledge(content, metadata={'source': 'manual'})
```

### Knowledge Base Files

Place `.txt` files in the knowledge directory:

```
knowledge/
├── face_swapping_guide.txt
├── gpu_optimization.txt
├── troubleshooting.txt
└── custom_tips.txt
```

## NVIDIA Optimizations

### TensorRT Integration

Enable TensorRT for faster inference:

```yaml
nvidia_config:
  tensorrt:
    enabled: true
    precision: "fp16"  # fp32, fp16, int8
    workspace_size: 1024  # MB
    cache_path: "./tensorrt_cache"
```

### CUDA Optimizations

```yaml
nvidia_config:
  cuda:
    memory_fraction: 0.9
    allow_growth: true
    streams: 2
```

### Performance Monitoring

```python
from roop.nvidia_optimizer import NVIDIAOptimizer

optimizer = NVIDIAOptimizer(settings)

# Get GPU information
gpu_info = optimizer.get_gpu_info()

# Benchmark performance
benchmark_results = optimizer.benchmark_operations()

# Get optimization recommendations
recommendations = optimizer.get_optimization_recommendations()
```

## Error Handling

### Enhanced Error Recovery

The system includes comprehensive error handling:

```python
from roop.error_handling import ErrorHandler, retry_on_error, error_context

# Initialize error handler
error_handler = ErrorHandler(settings)

# Automatic retry decorator
@retry_on_error(max_retries=3, delay=1.0)
def risky_operation():
    # Code that might fail
    pass

# Error context manager
with error_context("face_processing", error_handler):
    # Code that might fail
    process_face()
```

### Error Configuration

```yaml
error_handling:
  max_retries: 3
  retry_delay: 1.0
  fallback_to_cpu: true
  detailed_logging: true
  crash_recovery: true

logging:
  level: "INFO"
  file_path: "./logs/roop.log"
  detailed_errors: true
```

## API Reference

### Settings Class

```python
from settings import Settings

settings = Settings('config.yaml', 'config_defaults.yaml')

# Access AI settings
llm_enabled = settings.get_ai_setting('local_llm.enabled', False)
model_name = settings.get_ai_setting('local_llm.model', 'default')

# Access NVIDIA settings
tensorrt_enabled = settings.get_nvidia_setting('tensorrt.enabled', False)

# Access agent settings
rag_enabled = settings.get_agent_setting('rag_agent', 'enabled', False)

# Validate configuration
is_valid = settings.validate()

# Save configuration
settings.save()
```

### LLM Integration

```python
from roop.llm_integration import LLMManager

llm_manager = LLMManager(settings)

if llm_manager.is_available():
    response = llm_manager.generate("How to improve face swapping quality?")
    
    # Streaming response
    for chunk in llm_manager.generate_stream("Explain face detection"):
        print(chunk, end='')
    
    # Health check
    health = llm_manager.health_check()
```

### RAG System

```python
from roop.rag_system import RAGSystem

rag_system = RAGSystem(settings, llm_manager)

# Query knowledge base
answer = rag_system.query("What are the best face swapping practices?")

# Add knowledge
success = rag_system.add_knowledge("New information about face processing")

# Get statistics
stats = rag_system.get_stats()
```

## Troubleshooting

### Common Issues

1. **LLM not available**: Check Ollama installation and model download
2. **RAG not working**: Ensure sentence-transformers is installed
3. **TensorRT errors**: Verify TensorRT installation and CUDA compatibility
4. **Memory issues**: Adjust CUDA memory fraction in configuration

### Debug Mode

Enable detailed logging:

```yaml
logging:
  level: "DEBUG"
  detailed_errors: true
  console_output: true
```

### Health Checks

```python
# System status
status = manager.get_system_status()
print(f"Agents: {status['agent_count']}")
print(f"LLM Status: {status.get('llm_status', 'Not available')}")

# NVIDIA status
nvidia_status = optimizer.get_status()
print(f"GPU Available: {nvidia_status['gpu_info']['cuda_available']}")
```

## Migration Guide

### From Basic to Enhanced

1. **Backup current configuration**:
   ```bash
   cp config.yaml config.yaml.backup
   ```

2. **Install enhanced dependencies**:
   ```bash
   pip install -r requirements-ai.txt
   ```

3. **Run migration**:
   ```bash
   python -c "from roop.config_migrator import migrate_config; migrate_config()"
   ```

4. **Enable desired features** in the new configuration

5. **Test functionality** with the enhanced agent system

### Gradual Adoption

The enhanced features are designed for gradual adoption:

- Basic functionality remains unchanged
- Enhanced features gracefully degrade if dependencies are missing
- Configuration is backward compatible
- Existing workflows continue to work

## Performance Considerations

### Memory Usage

Enhanced features may increase memory usage:

- RAG embeddings: ~100MB for default knowledge base
- LLM integration: Depends on model size
- TensorRT cache: ~50-500MB per optimized model

### Optimization Tips

1. **Use TensorRT** for production deployments
2. **Enable CUDA streams** for parallel processing
3. **Adjust memory fractions** based on available VRAM
4. **Use FP16 precision** for faster inference
5. **Cache embeddings** for frequently used knowledge

## Contributing

To contribute to the enhanced AI features:

1. Follow the existing code style and patterns
2. Add comprehensive error handling
3. Include tests for new functionality
4. Update documentation
5. Ensure graceful degradation when dependencies are missing

## License

Enhanced features maintain the same license as the base Roop Unleashed project.