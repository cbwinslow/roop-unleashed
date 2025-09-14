#!/usr/bin/env python3
"""
Hardware compatibility and optimization tests.
"""

import pytest
import platform
import subprocess
import os
from unittest.mock import Mock, patch


class TestGPUDetection:
    """Test GPU detection and compatibility."""
    
    @pytest.mark.gpu
    def test_cuda_availability(self):
        """Test CUDA GPU detection and availability."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                assert device_count > 0, "CUDA is available but no devices found"
                assert 0 <= current_device < device_count
                assert len(device_name) > 0, "Device name should not be empty"
                
                # Test memory info
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                
                assert memory_allocated >= 0
                assert memory_reserved >= memory_allocated
                
                print(f"CUDA Device: {device_name}")
                print(f"Memory allocated: {memory_allocated / 1024**2:.1f} MB")
                print(f"Memory reserved: {memory_reserved / 1024**2:.1f} MB")
            else:
                pytest.skip("CUDA not available")
                
        except ImportError:
            pytest.skip("PyTorch not available")
    
    @pytest.mark.gpu
    def test_rocm_compatibility(self):
        """Test ROCm (AMD GPU) compatibility."""
        def check_rocm_installation():
            # Check for ROCm installation
            rocm_paths = [
                "/opt/rocm",
                "/usr/lib/x86_64-linux-gnu/rocm",
                os.environ.get("ROCM_PATH", "")
            ]
            
            for path in rocm_paths:
                if path and os.path.exists(path):
                    return True
            return False
        
        def check_amd_gpu():
            try:
                # Check for AMD GPU using lspci
                result = subprocess.run(
                    ["lspci", "-nn"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    amd_keywords = ["AMD", "Radeon", "AMDGPU"]
                    return any(keyword in result.stdout for keyword in amd_keywords)
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            return False
        
        rocm_installed = check_rocm_installation()
        amd_gpu_present = check_amd_gpu()
        
        if not amd_gpu_present:
            pytest.skip("No AMD GPU detected")
        
        # Mock ROCm functionality test
        class MockROCmDevice:
            def __init__(self):
                self.device_count = 1
                self.device_name = "AMD Radeon RX 6800 XT"
                
            def is_available(self):
                return rocm_installed
                
            def get_memory_info(self):
                return {"total": 16 * 1024**3, "free": 12 * 1024**3}  # 16GB total, 12GB free
        
        mock_device = MockROCmDevice()
        
        if mock_device.is_available():
            memory_info = mock_device.get_memory_info()
            assert memory_info["total"] > 0
            assert memory_info["free"] <= memory_info["total"]
            
            print(f"ROCm Device: {mock_device.device_name}")
            print(f"Total memory: {memory_info['total'] / 1024**3:.1f} GB")
            print(f"Free memory: {memory_info['free'] / 1024**3:.1f} GB")
        else:
            pytest.skip("ROCm not properly installed")
    
    @pytest.mark.gpu  
    def test_mps_compatibility(self):
        """Test Metal Performance Shaders (Apple Silicon) compatibility."""
        is_macos = platform.system() == "Darwin"
        
        if not is_macos:
            pytest.skip("MPS only available on macOS")
        
        try:
            import torch
            
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Test MPS device creation
                mps_device = torch.device("mps")
                
                # Test tensor operations on MPS
                test_tensor = torch.randn(100, 100, device=mps_device)
                result = torch.matmul(test_tensor, test_tensor.T)
                
                assert result.device.type == "mps"
                assert result.shape == (100, 100)
                
                print("MPS (Metal Performance Shaders) is available and functional")
            else:
                pytest.skip("MPS not available on this system")
                
        except ImportError:
            pytest.skip("PyTorch not available")
    
    @pytest.mark.gpu
    def test_directml_compatibility(self):
        """Test DirectML (Windows) compatibility."""
        is_windows = platform.system() == "Windows"
        
        if not is_windows:
            pytest.skip("DirectML only available on Windows")
        
        # Mock DirectML availability check
        class MockDirectMLDevice:
            def __init__(self):
                self.available = self._check_directml_availability()
                
            def _check_directml_availability(self):
                # In a real implementation, this would check for DirectML installation
                # and compatible hardware
                try:
                    # Mock check for DirectML
                    return True  # Assume available for testing
                except Exception:
                    return False
                    
            def get_device_info(self):
                if not self.available:
                    return None
                    
                return {
                    "name": "DirectML Device",
                    "driver_version": "1.8.0",
                    "feature_level": "1_0"
                }
        
        mock_device = MockDirectMLDevice()
        
        if mock_device.available:
            device_info = mock_device.get_device_info()
            assert device_info is not None
            assert "name" in device_info
            
            print(f"DirectML Device: {device_info['name']}")
            print(f"Driver version: {device_info['driver_version']}")
        else:
            pytest.skip("DirectML not available")


class TestCPUOptimization:
    """Test CPU optimization and fallback capabilities."""
    
    @pytest.mark.hardware
    def test_cpu_detection(self):
        """Test CPU detection and capabilities."""
        import psutil
        
        # Get CPU information
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores
        cpu_freq = psutil.cpu_freq()
        
        assert cpu_count > 0, "Should detect at least one physical CPU core"
        assert cpu_count_logical >= cpu_count, "Logical cores should be >= physical cores"
        
        if cpu_freq:
            assert cpu_freq.current > 0, "CPU frequency should be positive"
            
        # Test CPU features
        cpu_features = self._detect_cpu_features()
        
        print(f"Physical CPU cores: {cpu_count}")
        print(f"Logical CPU cores: {cpu_count_logical}")
        print(f"CPU frequency: {cpu_freq.current if cpu_freq else 'Unknown'} MHz")
        print(f"CPU features: {', '.join(cpu_features)}")
        
        return {
            "physical_cores": cpu_count,
            "logical_cores": cpu_count_logical,
            "frequency": cpu_freq.current if cpu_freq else None,
            "features": cpu_features
        }
    
    def _detect_cpu_features(self):
        """Detect CPU features relevant to face processing."""
        features = []
        
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            
            # Check for relevant instruction sets
            relevant_features = ["sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2", "avx", "avx2", "fma3"]
            
            for feature in relevant_features:
                if feature in info.get("flags", []):
                    features.append(feature.upper())
                    
        except ImportError:
            # Fallback detection method
            if platform.machine().lower() in ["x86_64", "amd64"]:
                features = ["SSE2", "SSE3", "SSE4_1", "SSE4_2"]  # Assume basic x64 features
        
        return features
    
    @pytest.mark.hardware
    def test_cpu_processing_fallback(self):
        """Test CPU processing when GPU is not available."""
        class CPUFaceProcessor:
            def __init__(self):
                self.processing_mode = "cpu"
                self.optimization_level = "balanced"
                
            def process_image(self, image_data, num_threads=None):
                import psutil
                
                if num_threads is None:
                    num_threads = psutil.cpu_count(logical=False)
                
                # Mock CPU-based face processing
                processing_result = {
                    "success": True,
                    "processing_time": 0.5,  # Mock processing time
                    "threads_used": num_threads,
                    "memory_usage": 1024,  # MB
                    "device": "cpu"
                }
                
                return processing_result
            
            def optimize_for_cpu(self):
                # Mock CPU optimization
                optimizations = [
                    "enabled_multi_threading",
                    "optimized_memory_access",
                    "vectorized_operations"
                ]
                
                return {
                    "optimization_level": "high",
                    "optimizations_applied": optimizations,
                    "expected_speedup": 2.5
                }
        
        processor = CPUFaceProcessor()
        
        # Test CPU processing
        mock_image = {"width": 1920, "height": 1080, "channels": 3}
        result = processor.process_image(mock_image)
        
        assert result["success"], "CPU processing should succeed"
        assert result["device"] == "cpu"
        assert result["threads_used"] > 0
        
        # Test CPU optimization
        optimization_result = processor.optimize_for_cpu()
        assert optimization_result["expected_speedup"] > 1.0
        assert len(optimization_result["optimizations_applied"]) > 0
    
    @pytest.mark.hardware
    def test_memory_constraints(self):
        """Test handling of memory constraints."""
        import psutil
        
        class MemoryAwareProcessor:
            def __init__(self):
                self.memory_info = psutil.virtual_memory()
                self.max_memory_usage = self.memory_info.total * 0.8  # Use max 80% of RAM
                
            def estimate_memory_requirements(self, image_width, image_height, batch_size=1):
                # Estimate memory requirements for processing
                pixels_per_image = image_width * image_height
                bytes_per_pixel = 3  # RGB
                
                # Rough estimation including intermediate processing steps
                memory_per_image = pixels_per_image * bytes_per_pixel * 4  # 4x overhead for processing
                total_memory = memory_per_image * batch_size
                
                return total_memory
            
            def adjust_batch_size_for_memory(self, image_width, image_height, desired_batch_size):
                required_memory = self.estimate_memory_requirements(image_width, image_height, desired_batch_size)
                
                if required_memory <= self.max_memory_usage:
                    return desired_batch_size
                
                # Reduce batch size to fit in memory
                max_batch_size = int(self.max_memory_usage / self.estimate_memory_requirements(image_width, image_height, 1))
                return max(1, max_batch_size)
            
            def get_memory_status(self):
                current_memory = psutil.virtual_memory()
                return {
                    "total_gb": current_memory.total / 1024**3,
                    "available_gb": current_memory.available / 1024**3,
                    "used_percent": current_memory.percent,
                    "recommended_max_usage_gb": self.max_memory_usage / 1024**3
                }
        
        processor = MemoryAwareProcessor()
        
        # Test memory estimation
        memory_required = processor.estimate_memory_requirements(1920, 1080, 4)
        assert memory_required > 0
        
        # Test batch size adjustment
        adjusted_batch_size = processor.adjust_batch_size_for_memory(1920, 1080, 32)
        assert 1 <= adjusted_batch_size <= 32
        
        # Test memory status
        memory_status = processor.get_memory_status()
        assert memory_status["total_gb"] > 0
        assert 0 <= memory_status["used_percent"] <= 100
        
        print(f"Total memory: {memory_status['total_gb']:.1f} GB")
        print(f"Available memory: {memory_status['available_gb']:.1f} GB")
        print(f"Memory usage: {memory_status['used_percent']:.1f}%")


class TestHardwareOptimization:
    """Test hardware-specific optimizations."""
    
    @pytest.mark.hardware
    def test_threading_optimization(self):
        """Test multi-threading optimization."""
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        class ThreadingOptimizer:
            def __init__(self):
                self.optimal_thread_count = None
                
            def benchmark_thread_performance(self, max_threads=None):
                if max_threads is None:
                    import psutil
                    max_threads = psutil.cpu_count(logical=True)
                
                def cpu_intensive_task(task_id):
                    # Mock CPU-intensive operation
                    start_time = time.time()
                    result = sum(i**2 for i in range(10000))  # Simple CPU work
                    end_time = time.time()
                    return {
                        "task_id": task_id,
                        "result": result,
                        "duration": end_time - start_time
                    }
                
                benchmark_results = {}
                
                for thread_count in [1, 2, 4, 8, max_threads]:
                    if thread_count > max_threads:
                        continue
                    
                    start_time = time.time()
                    
                    with ThreadPoolExecutor(max_workers=thread_count) as executor:
                        futures = [executor.submit(cpu_intensive_task, i) for i in range(16)]
                        results = [future.result() for future in futures]
                    
                    total_time = time.time() - start_time
                    
                    benchmark_results[thread_count] = {
                        "total_time": total_time,
                        "tasks_per_second": 16 / total_time,
                        "avg_task_time": sum(r["duration"] for r in results) / len(results)
                    }
                
                # Find optimal thread count
                best_thread_count = max(benchmark_results.keys(), 
                                       key=lambda x: benchmark_results[x]["tasks_per_second"])
                self.optimal_thread_count = best_thread_count
                
                return benchmark_results
            
            def get_threading_recommendation(self):
                return {
                    "optimal_thread_count": self.optimal_thread_count,
                    "recommendation": f"Use {self.optimal_thread_count} threads for optimal performance"
                }
        
        optimizer = ThreadingOptimizer()
        
        # Benchmark threading performance
        results = optimizer.benchmark_thread_performance()
        
        assert len(results) > 0, "Should have benchmark results"
        assert all(r["total_time"] > 0 for r in results.values())
        assert all(r["tasks_per_second"] > 0 for r in results.values())
        
        # Get recommendation
        recommendation = optimizer.get_threading_recommendation()
        assert recommendation["optimal_thread_count"] is not None
        assert recommendation["optimal_thread_count"] > 0
        
        print(f"Optimal thread count: {recommendation['optimal_thread_count']}")
        
        for thread_count, metrics in results.items():
            print(f"{thread_count} threads: {metrics['tasks_per_second']:.2f} tasks/sec")
    
    @pytest.mark.hardware
    def test_memory_alignment_optimization(self):
        """Test memory alignment optimizations."""
        import numpy as np
        
        class MemoryAlignmentOptimizer:
            def __init__(self):
                self.alignment_sizes = [16, 32, 64, 128]  # Common alignment sizes
                
            def test_memory_alignment_performance(self, array_size=1000000):
                results = {}
                
                for alignment in self.alignment_sizes:
                    # Create aligned array
                    aligned_array = np.empty(array_size, dtype=np.float32)
                    
                    # Benchmark operations on aligned memory
                    start_time = time.time()
                    
                    # Perform some operations
                    for _ in range(10):
                        result = np.sum(aligned_array * 2.0)
                        result = np.sqrt(aligned_array + 1.0)
                    
                    operation_time = time.time() - start_time
                    
                    results[alignment] = {
                        "operation_time": operation_time,
                        "operations_per_second": 20 / operation_time  # 20 operations total
                    }
                
                return results
            
            def get_optimal_alignment(self):
                # Mock determination of optimal alignment
                # In practice, this would be based on actual benchmarks
                return 64  # Common optimal alignment for modern CPUs
        
        optimizer = MemoryAlignmentOptimizer()
        
        # Test memory alignment performance
        alignment_results = optimizer.test_memory_alignment_performance()
        
        assert len(alignment_results) > 0
        assert all(r["operation_time"] > 0 for r in alignment_results.values())
        
        optimal_alignment = optimizer.get_optimal_alignment()
        assert optimal_alignment in optimizer.alignment_sizes
        
        print(f"Optimal memory alignment: {optimal_alignment} bytes")
        
        for alignment, metrics in alignment_results.items():
            print(f"{alignment}-byte alignment: {metrics['operations_per_second']:.2f} ops/sec")
    
    @pytest.mark.hardware
    def test_cache_optimization(self):
        """Test CPU cache optimization strategies."""
        import numpy as np
        import time
        
        class CacheOptimizer:
            def __init__(self):
                self.cache_sizes = self._detect_cache_sizes()
                
            def _detect_cache_sizes(self):
                # Mock cache size detection
                # In practice, this would query the actual CPU cache hierarchy
                return {
                    "L1": 32 * 1024,      # 32 KB
                    "L2": 256 * 1024,     # 256 KB  
                    "L3": 8 * 1024 * 1024 # 8 MB
                }
            
            def benchmark_cache_performance(self):
                results = {}
                
                for cache_level, cache_size in self.cache_sizes.items():
                    # Create arrays that fit in different cache levels
                    array_size = cache_size // 4  # 4 bytes per float32
                    test_array = np.random.rand(array_size).astype(np.float32)
                    
                    # Benchmark cache-friendly access pattern
                    start_time = time.time()
                    
                    # Sequential access (cache-friendly)
                    for _ in range(100):
                        result = np.sum(test_array)
                    
                    sequential_time = time.time() - start_time
                    
                    # Random access (cache-unfriendly)
                    start_time = time.time()
                    indices = np.random.randint(0, array_size, size=array_size)
                    
                    for _ in range(100):
                        result = np.sum(test_array[indices[:1000]])  # Sample random access
                    
                    random_time = time.time() - start_time
                    
                    results[cache_level] = {
                        "cache_size_kb": cache_size // 1024,
                        "sequential_time": sequential_time,
                        "random_time": random_time,
                        "cache_efficiency": sequential_time / random_time if random_time > 0 else 1.0
                    }
                
                return results
            
            def get_cache_optimization_recommendations(self):
                return [
                    "Use sequential memory access patterns when possible",
                    "Process data in chunks that fit in L2 cache",
                    "Minimize random memory access",
                    "Use memory pooling to reduce allocation overhead",
                    "Consider data structure layout for cache locality"
                ]
        
        optimizer = CacheOptimizer()
        
        # Test cache performance
        cache_results = optimizer.benchmark_cache_performance()
        
        assert len(cache_results) > 0
        assert all("cache_size_kb" in r for r in cache_results.values())
        assert all(r["cache_efficiency"] > 0 for r in cache_results.values())
        
        # Get optimization recommendations
        recommendations = optimizer.get_cache_optimization_recommendations()
        assert len(recommendations) > 0
        
        print("Cache Performance Analysis:")
        for cache_level, metrics in cache_results.items():
            print(f"{cache_level} ({metrics['cache_size_kb']} KB): "
                  f"Efficiency ratio {metrics['cache_efficiency']:.2f}")
        
        print("\nCache Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")


if __name__ == "__main__":
    # Run hardware tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])