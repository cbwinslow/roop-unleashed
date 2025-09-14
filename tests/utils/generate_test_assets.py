#!/usr/bin/env python3
"""
Generate test assets for CI workflow.
"""

import os
from PIL import Image
import numpy as np


def create_test_image(path, width=256, height=256):
    """Create a simple test image."""
    # Create a simple gradient image
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient pattern
    for y in range(height):
        for x in range(width):
            image_array[y, x] = [
                int(255 * x / width),  # Red gradient
                int(255 * y / height),  # Green gradient
                128  # Blue constant
            ]
    
    image = Image.fromarray(image_array)
    image.save(path)
    print(f"Created test image: {path}")


def generate_test_assets():
    """Generate simple test assets for CI."""
    os.makedirs("test-assets", exist_ok=True)
    
    # Create simple test images
    create_test_image("test-assets/test_source.jpg")
    create_test_image("test-assets/test_target.jpg", 512, 512)
    
    print("Test assets generated successfully")


if __name__ == "__main__":
    try:
        generate_test_assets()
    except ImportError as e:
        print(f"Missing dependency for test asset generation: {e}")
        print("Creating placeholder files instead...")
        os.makedirs("test-assets", exist_ok=True)
        with open("test-assets/test_source.jpg", "w") as f:
            f.write("# Placeholder test file\n")
        with open("test-assets/test_target.jpg", "w") as f:
            f.write("# Placeholder test file\n")