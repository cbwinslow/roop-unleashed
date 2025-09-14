"""
NVIDIA optimization utilities for roop-unleashed.
Provides TensorRT integration, CUDA optimizations, and performance monitoring.
"""

import os
import logging
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import subprocess

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from roop.error_handling import RoopException, retry_on_error, safe_execute

logger = logging.getLogger(__name__)


class NVIDIAOptimizationError(RoopException):
    """Raised when NVIDIA optimization operations fail."""
    pass


class CUDAProfiler:
    """CUDA performance profiler and monitor."""
    
    def __init__(self):
        self.enabled = TORCH_AVAILABLE and torch.cuda.is_available()
        self.profile_data = {}
        self.start_times = {}
    
    def start_profiling(self, operation_name: str):
        """Start profiling a CUDA operation."""
        if not self.enabled:
            return
        
        torch.cuda.synchronize()
        self.start_times[operation_name] = time.time()
    
    def end_profiling(self, operation_name: str):
        """End profiling and record results."""
        if not self.enabled or operation_name not in self.start_times:
            return
        
        torch.cuda.synchronize()
        duration = time.time() - self.start_times[operation_name]
        
        if operation_name not in self.profile_data:
            self.profile_data[operation_name] = []
        
        self.profile_data[operation_name].append(duration)
        del self.start_times[operation_name]
        
        logger.debug(f"CUDA operation '{operation_name}' took {duration:.4f}s")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get CUDA memory information."""
        if not self.enabled:
            return {'available': False}
        
        try:
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            memory_free, memory_total = torch.cuda.mem_get_info()
            
            return {
                'available': True,
                'allocated_mb': memory_allocated / 1024 / 1024,
                'reserved_mb': memory_reserved / 1024 / 1024,
                'free_mb': memory_free / 1024 / 1024,
                'total_mb': memory_total / 1024 / 1024,
                'utilization_percent': (memory_allocated / memory_total) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get CUDA memory info: {e}")
            return {'available': False, 'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        stats = {}
        for operation, durations in self.profile_data.items():
            if durations:
                stats[operation] = {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'average_time': sum(durations) / len(durations),
                    'min_time': min(durations),
                    'max_time': max(durations)
                }
        return stats
    
    def clear_stats(self):
        """Clear profiling statistics."""
        self.profile_data.clear()
        self.start_times.clear()


class TensorRTOptimizer:
    """TensorRT model optimization and inference."""
    
    def __init__(self, settings):
        self.settings = settings
        self.enabled = (TENSORRT_AVAILABLE and 
                       settings.get_nvidia_setting('tensorrt.enabled', False))
        self.precision = settings.get_nvidia_setting('tensorrt.precision', 'fp16')
        self.workspace_size = settings.get_nvidia_setting('tensorrt.workspace_size', 1024) * 1024 * 1024
        self.cache_path = Path(settings.get_nvidia_setting('tensorrt.cache_path', './tensorrt_cache'))
        
        if self.enabled:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            logger.info("TensorRT optimizer initialized")
    
    def optimize_onnx_model(self, onnx_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """Optimize an ONNX model using TensorRT."""
        if not self.enabled:
            logger.warning("TensorRT optimization is disabled")
            return None
        
        try:
            # Generate cache filename
            if output_path is None:
                model_name = Path(onnx_path).stem
                output_path = str(self.cache_path / f"{model_name}_{self.precision}.trt")
            
            # Check if optimized model already exists
            if os.path.exists(output_path):
                logger.info(f"Using cached TensorRT model: {output_path}")
                return output_path
            
            logger.info(f"Optimizing ONNX model {onnx_path} with TensorRT...")
            
            # Create TensorRT logger and builder
            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)
            config = builder.create_builder_config()
            
            # Set precision
            if self.precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            elif self.precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Set workspace size
            config.max_workspace_size = self.workspace_size
            
            # Parse ONNX model
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)
            
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"TensorRT parsing error: {parser.get_error(error)}")
                    return None
            
            # Build engine
            engine = builder.build_engine(network, config)
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return None
            
            # Serialize and save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT optimization completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return None
    
    def create_inference_context(self, engine_path: str):
        """Create a TensorRT inference context."""
        if not self.enabled:
            return None
        
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            return context
        except Exception as e:
            logger.error(f"Failed to create TensorRT context: {e}")
            return None


class CUDAStreamManager:
    """Manages CUDA streams for parallel processing."""
    
    def __init__(self, settings):
        self.settings = settings
        self.enabled = TORCH_AVAILABLE and torch.cuda.is_available()
        self.stream_count = settings.get_nvidia_setting('cuda.streams', 2)
        self.streams = []
        
        if self.enabled:
            self._initialize_streams()
    
    def _initialize_streams(self):
        """Initialize CUDA streams."""
        try:
            for i in range(self.stream_count):
                stream = torch.cuda.Stream()
                self.streams.append(stream)
            
            logger.info(f"Initialized {len(self.streams)} CUDA streams")
        except Exception as e:
            logger.error(f"Failed to initialize CUDA streams: {e}")
            self.enabled = False
    
    def get_stream(self, index: int = 0):
        """Get a CUDA stream by index."""
        if not self.enabled or not self.streams:
            return None
        
        return self.streams[index % len(self.streams)]
    
    def synchronize_all(self):
        """Synchronize all streams."""
        if not self.enabled:
            return
        
        for stream in self.streams:
            stream.synchronize()


class NVIDIAOptimizer:
    """Main NVIDIA optimization coordinator."""
    
    def __init__(self, settings):
        self.settings = settings
        self.profiler = CUDAProfiler()
        self.tensorrt = TensorRTOptimizer(settings)
        self.stream_manager = CUDAStreamManager(settings)
        
        # Apply CUDA optimizations
        self._apply_cuda_optimizations()
    
    def _apply_cuda_optimizations(self):
        """Apply various CUDA optimizations."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        try:
            # Set memory fraction
            memory_fraction = self.settings.get_nvidia_setting('cuda.memory_fraction', 0.9)
            if memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # Enable memory growth if specified
            if self.settings.get_nvidia_setting('cuda.allow_growth', True):
                # PyTorch doesn't have direct equivalent, but we can use memory caching
                torch.cuda.empty_cache()
            
            # Set CUDA optimization flags
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches
            os.environ['CUDA_CACHE_DISABLE'] = '0'    # Enable CUDA caching
            
            logger.info("Applied CUDA optimizations")
            
        except Exception as e:
            logger.error(f"Failed to apply CUDA optimizations: {e}")
    
    def optimize_model_loading(self, model_path: str) -> Optional[str]:
        """Optimize model loading with TensorRT if available."""
        if not self.tensorrt.enabled:
            return None
        
        if model_path.endswith('.onnx'):
            return self.tensorrt.optimize_onnx_model(model_path)
        
        return None
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        info = {
            'cuda_available': False,
            'tensorrt_available': TENSORRT_AVAILABLE,
            'cupy_available': CUPY_AVAILABLE,
            'devices': []
        }
        
        if not TORCH_AVAILABLE:
            return info
        
        try:
            info['cuda_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                info['cuda_version'] = torch.version.cuda
                info['device_count'] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    device_info = {
                        'id': i,
                        'name': device_props.name,
                        'major': device_props.major,
                        'minor': device_props.minor,
                        'total_memory': device_props.total_memory,
                        'multi_processor_count': device_props.multi_processor_count
                    }
                    info['devices'].append(device_info)
                
                # Add memory info for current device
                memory_info = self.profiler.get_memory_info()
                info['memory'] = memory_info
        
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            info['error'] = str(e)
        
        return info
    
    def benchmark_operations(self) -> Dict[str, Any]:
        """Benchmark common operations to assess performance."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'available': False}
        
        results = {'available': True, 'benchmarks': {}}
        
        try:
            device = torch.cuda.current_device()
            
            # Matrix multiplication benchmark
            size = 1000
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warm up
            for _ in range(5):
                torch.mm(a, b)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):
                result = torch.mm(a, b)
            
            torch.cuda.synchronize()
            matmul_time = (time.time() - start_time) / 10
            
            results['benchmarks']['matrix_multiply_1000x1000'] = {
                'time_seconds': matmul_time,
                'gflops': (2 * size**3) / (matmul_time * 1e9)
            }
            
            # Memory bandwidth test
            size_mb = 100
            data = torch.randn(size_mb * 1024 * 1024 // 4, device=device)  # 100MB
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):
                data_copy = data.clone()
            
            torch.cuda.synchronize()
            copy_time = (time.time() - start_time) / 10
            
            results['benchmarks']['memory_copy_100mb'] = {
                'time_seconds': copy_time,
                'bandwidth_gb_s': (size_mb / 1024) / copy_time
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def optimize_for_inference(self):
        """Apply optimizations specifically for inference."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # Set inference mode optimizations
            torch.set_grad_enabled(False)
            
            # Enable cuDNN benchmarking for consistent input sizes
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            logger.info("Applied inference optimizations")
            
        except Exception as e:
            logger.error(f"Failed to apply inference optimizations: {e}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current setup."""
        recommendations = []
        
        gpu_info = self.get_gpu_info()
        
        if not gpu_info['cuda_available']:
            recommendations.append("CUDA is not available. Consider installing CUDA drivers for GPU acceleration.")
            return recommendations
        
        # Check memory
        memory_info = gpu_info.get('memory', {})
        if memory_info.get('utilization_percent', 0) > 90:
            recommendations.append("GPU memory utilization is high. Consider reducing batch size or frame buffer size.")
        
        # Check TensorRT
        if not TENSORRT_AVAILABLE:
            recommendations.append("TensorRT is not available. Install TensorRT for faster inference.")
        elif not self.tensorrt.enabled:
            recommendations.append("TensorRT optimization is disabled. Enable it in configuration for better performance.")
        
        # Check CUDA streams
        if self.stream_manager.stream_count < 2:
            recommendations.append("Consider using multiple CUDA streams for better parallelization.")
        
        # Check precision
        if self.tensorrt.precision == 'fp32':
            recommendations.append("Consider using FP16 precision for faster inference with minimal quality loss.")
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        return {
            'gpu_info': self.get_gpu_info(),
            'tensorrt_enabled': self.tensorrt.enabled,
            'cuda_streams': len(self.stream_manager.streams),
            'profiler_stats': self.profiler.get_stats(),
            'recommendations': self.get_optimization_recommendations(),
            'settings': {
                'memory_fraction': self.settings.get_nvidia_setting('cuda.memory_fraction', 0.9),
                'tensorrt_precision': self.tensorrt.precision,
                'workspace_size_mb': self.settings.get_nvidia_setting('tensorrt.workspace_size', 1024)
            }
        }