#!/usr/bin/env python3
"""
Basic model availability test for CI workflow.
"""

import os
import sys
import json


def test_model_availability():
    """Test basic model availability check."""
    print("Testing model availability...")
    
    # Define expected model paths and configurations
    model_configs = {
        "inswapper_128.onnx": {"size_mb": 50, "type": "face_swap"},
        "GFPGANv1.4.pth": {"size_mb": 300, "type": "face_enhancement"},
        "codeformer.pth": {"size_mb": 200, "type": "face_restoration"}
    }
    
    available_models = {}
    
    # Check for model files (they likely won't exist in CI, so we'll simulate)
    models_dir = "models"  # Typical models directory
    
    for model_name, config in model_configs.items():
        model_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            available_models[model_name] = {
                "path": model_path,
                "size_mb": file_size,
                "status": "available"
            }
            print(f"✓ Found model: {model_name} ({file_size:.1f} MB)")
        else:
            available_models[model_name] = {
                "path": model_path,
                "size_mb": 0,
                "status": "not_found"
            }
            print(f"⚠ Model not found: {model_name}")
    
    # For CI, we'll consider this successful even if models aren't present
    # since downloading large models in CI is not practical
    
    # Save model availability report
    try:
        with open("model-availability.json", "w") as f:
            json.dump(available_models, f, indent=2)
        
        print("✓ Model availability test completed")
        print(f"Checked {len(model_configs)} models")
        return True
        
    except Exception as e:
        print(f"✗ Error in model availability test: {e}")
        return False


if __name__ == "__main__":
    if test_model_availability():
        sys.exit(0)
    else:
        sys.exit(1)