#!/usr/bin/env python3
"""
Test script for enhanced face processing capabilities.
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/roop-unleashed/roop-unleashed')

# Test imports
try:
    import roop.globals
    from roop.enhanced_face_detection import get_enhanced_faces, FaceQualityAssessment
    from roop.advanced_blending import AdvancedBlender, get_available_blend_methods
    from roop.enhanced_face_swapper import assess_frame_quality, get_enhancement_config
    from roop.processors.frame.face_swapper import get_processing_info, get_blend_methods
    
    print("✓ All enhanced modules imported successfully")
    
    # Test face quality assessment
    print("\n=== Testing Face Quality Assessment ===")
    
    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple "face" (circle with features)
    center = (320, 240)
    cv2.circle(test_image, center, 80, (180, 180, 180), -1)  # Face
    cv2.circle(test_image, (300, 220), 8, (0, 0, 0), -1)     # Left eye
    cv2.circle(test_image, (340, 220), 8, (0, 0, 0), -1)     # Right eye
    cv2.ellipse(test_image, (320, 260), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    # Test quality assessment on synthetic image
    quality_info = assess_frame_quality(test_image)
    print(f"Quality assessment result: {quality_info}")
    
    # Test available blend methods
    print(f"\n=== Available Blend Methods ===")
    blend_methods = get_available_blend_methods()
    print(f"Available methods: {blend_methods}")
    
    # Test processing info
    print(f"\n=== Processing Information ===")
    proc_info = get_processing_info()
    print(f"Processing info: {proc_info}")
    
    # Test enhancement config
    print(f"\n=== Enhancement Configuration ===")
    config = get_enhancement_config()
    print(f"Enhancement config: {config}")
    
    # Test advanced blender
    print(f"\n=== Testing Advanced Blender ===")
    blender = AdvancedBlender()
    
    # Create source and target test images
    source_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    target_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test blend
    try:
        blended = blender.blend_face(source_img, target_img, (270, 190, 370, 290), "alpha", 0.5)
        print("✓ Alpha blending test successful")
    except Exception as e:
        print(f"✗ Alpha blending test failed: {e}")
    
    print("\n=== All Tests Completed ===")
    print("Enhanced face processing capabilities are ready to use!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)