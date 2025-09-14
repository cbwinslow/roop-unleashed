#!/usr/bin/env python3
"""
Pytest configuration and fixtures for the roop-unleashed test suite.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_OUTPUT_DIR = Path(__file__).parent / "outputs"


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    return TEST_DATA_DIR


@pytest.fixture(scope="session") 
def test_output_dir():
    """Provide path for test outputs."""
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    return TEST_OUTPUT_DIR


@pytest.fixture
def sample_face_image():
    """Generate a sample face image for testing."""
    # Create a simple synthetic face image
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    # Add some structure to make it more face-like
    img[100:400, 100:400] = 150  # Face region
    img[150:180, 200:250] = 50   # Eyes
    img[150:180, 350:400] = 50   
    img[250:280, 225:325] = 100  # Nose
    img[320:350, 200:350] = 80   # Mouth
    return Image.fromarray(img)


@pytest.fixture
def sample_target_image():
    """Generate a sample target image for testing."""
    img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    # Add different face structure
    img[80:380, 150:450] = 180   # Face region
    img[120:150, 220:270] = 30   # Eyes
    img[120:150, 350:400] = 30   
    img[200:230, 260:340] = 120  # Nose
    img[280:310, 240:380] = 60   # Mouth
    return Image.fromarray(img)


@pytest.fixture
def temp_image_file(tmp_path, sample_face_image):
    """Create a temporary image file."""
    image_path = tmp_path / "test_image.jpg"
    sample_face_image.save(image_path)
    return str(image_path)


@pytest.fixture
def temp_video_file(tmp_path):
    """Create a temporary video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    # Note: This is a placeholder - in real tests, you'd create an actual video
    video_path.write_text("dummy video content")
    return str(video_path)


@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
            
        def start_timer(self, name):
            import time
            self.metrics[f"{name}_start"] = time.time()
            
        def end_timer(self, name):
            import time
            if f"{name}_start" in self.metrics:
                self.metrics[f"{name}_duration"] = time.time() - self.metrics[f"{name}_start"]
                
        def get_metrics(self):
            return self.metrics.copy()
    
    return PerformanceTracker()


@pytest.fixture
def quality_metrics():
    """Provide quality assessment tools."""
    class QualityMetrics:
        @staticmethod
        def ssim(img1, img2):
            """Calculate SSIM between two images."""
            try:
                from skimage.metrics import structural_similarity as ssim
                import cv2
                
                # Convert PIL images to numpy arrays
                if hasattr(img1, 'save'):  # PIL Image
                    img1 = np.array(img1)
                if hasattr(img2, 'save'):  # PIL Image
                    img2 = np.array(img2)
                    
                # Convert to grayscale for SSIM
                gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                
                return ssim(gray1, gray2)
            except ImportError:
                # Fallback implementation
                return 0.8  # Mock value for testing
        
        @staticmethod
        def psnr(img1, img2):
            """Calculate PSNR between two images."""
            try:
                import cv2
                
                if hasattr(img1, 'save'):  # PIL Image
                    img1 = np.array(img1)
                if hasattr(img2, 'save'):  # PIL Image
                    img2 = np.array(img2)
                    
                return cv2.PSNR(img1, img2)
            except ImportError:
                # Fallback implementation  
                mse = np.mean((img1 - img2) ** 2)
                if mse == 0:
                    return float('inf')
                return 20 * np.log10(255.0 / np.sqrt(mse))
        
        @staticmethod
        def face_similarity(face1, face2):
            """Calculate face similarity score."""
            # Placeholder implementation
            return 0.85
    
    return QualityMetrics()


@pytest.fixture(scope="session")
def test_models_dir():
    """Provide directory for test models."""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    class MemoryMonitor:
        def __init__(self):
            self.initial_memory = self.get_memory_usage()
            
        def get_memory_usage(self):
            """Get current memory usage in MB."""
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except ImportError:
                return 0
                
        def get_memory_delta(self):
            """Get memory usage change since initialization."""
            return self.get_memory_usage() - self.initial_memory
    
    return MemoryMonitor()


# Test markers for different categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "agents: AI agent tests")
    config.addinivalue_line("markers", "face_processing: Face processing tests")


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip GPU tests if no GPU available
    if "gpu" in item.keywords:
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after tests."""
    yield
    # Cleanup code would go here
    pass


@pytest.fixture
def error_tracker():
    """Track errors and exceptions during tests."""
    class ErrorTracker:
        def __init__(self):
            self.errors = []
            
        def add_error(self, error_type, message, details=None):
            self.errors.append({
                'type': error_type,
                'message': message,
                'details': details,
                'timestamp': __import__('time').time()
            })
            
        def get_errors(self):
            return self.errors.copy()
            
        def clear_errors(self):
            self.errors.clear()
    
    return ErrorTracker()