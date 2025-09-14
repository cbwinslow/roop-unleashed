#!/usr/bin/env python3
"""
Simple memory profiling test for CI compatibility.
"""

import sys
import psutil
import os


def test_memory_usage():
    """Basic memory usage test that doesn't require heavy dependencies."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Check that memory usage is reasonable (less than 1GB for basic operations)
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")
    
    # Write basic memory profile
    with open("memory-profile.txt", "w") as f:
        f.write(f"Memory usage: {memory_mb:.2f} MB\n")
        f.write(f"Virtual memory: {memory_info.vms / 1024 / 1024:.2f} MB\n")
    
    assert memory_mb < 1024, f"Memory usage too high: {memory_mb} MB"
    print("Memory test passed")


if __name__ == "__main__":
    test_memory_usage()