#!/usr/bin/env python3
"""
Basic CPU processing test for CI workflow.
"""

import time
import sys


def test_cpu_processing():
    """Test basic CPU processing capabilities."""
    print("Testing CPU processing...")
    
    # Simple CPU-bound task
    start_time = time.time()
    
    # Basic computation test
    result = 0
    for i in range(100000):
        result += i * i
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"CPU processing test completed in {processing_time:.3f} seconds")
    print(f"Result: {result}")
    
    # Validate that processing completed in reasonable time
    if processing_time > 10.0:  # Should complete in under 10 seconds
        print("⚠ CPU processing took longer than expected")
        return False
    
    print("✓ CPU processing test passed")
    return True


if __name__ == "__main__":
    if test_cpu_processing():
        sys.exit(0)
    else:
        sys.exit(1)