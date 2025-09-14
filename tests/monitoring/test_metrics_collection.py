#!/usr/bin/env python3
"""
Basic metrics collection test for CI workflow.
"""

import json
import time
import sys


def test_metrics_collection():
    """Test basic metrics collection."""
    print("Testing metrics collection...")
    
    # Simulate collecting basic metrics
    metrics = {
        "timestamp": time.time(),
        "system": {
            "cpu_usage": 25.5,  # Simulated values
            "memory_usage": 45.2,
            "disk_usage": 30.1
        },
        "application": {
            "requests_per_second": 10.0,
            "response_time_ms": 150.0,
            "error_rate": 0.01
        },
        "status": "healthy"
    }
    
    # Write metrics to file
    try:
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("✓ Metrics collected and saved successfully")
        print(f"Sample metrics: {json.dumps(metrics['system'], indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error collecting metrics: {e}")
        return False


if __name__ == "__main__":
    if test_metrics_collection():
        sys.exit(0)
    else:
        sys.exit(1)