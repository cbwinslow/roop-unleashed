#!/usr/bin/env python3
"""
Basic memory limits test for CI workflow.
"""

import sys
import psutil
import os


def test_memory_limits():
    """Test memory usage limits."""
    print("Testing memory limits...")
    
    # Get current memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")
    
    # Test creating some data structures
    test_data = []
    try:
        # Create some test data (small scale for CI)
        for i in range(1000):
            test_data.append([j for j in range(100)])
        
        # Check memory again
        new_memory_info = process.memory_info()
        new_memory_mb = new_memory_info.rss / 1024 / 1024
        
        print(f"Memory after test data creation: {new_memory_mb:.2f} MB")
        print(f"Memory increase: {new_memory_mb - memory_mb:.2f} MB")
        
        # Clean up
        test_data.clear()
        
        # Validate memory usage is reasonable for CI
        if new_memory_mb > 2048:  # 2GB limit for CI
            print(f"⚠ Memory usage too high: {new_memory_mb:.2f} MB")
            return False
        
        print("✓ Memory limits test passed")
        return True
        
    except MemoryError:
        print("✗ Memory error occurred during test")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    if test_memory_limits():
        sys.exit(0)
    else:
        sys.exit(1)