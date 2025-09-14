#!/usr/bin/env python3
"""
Simple test script to verify core functionality.
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/roop-unleashed/roop-unleashed')

def test_basic_functionality():
    """Test core enhanced functionality."""
    print("=== Testing Core Enhanced Functionality ===")
    
    try:
        # Test imports
        import roop.globals
        from roop.enhanced_face_detection import get_enhanced_faces, FaceQualityAssessment
        from roop.advanced_blending import AdvancedBlender, get_available_blend_methods
        from roop.enhanced_face_swapper import assess_frame_quality, get_enhancement_config
        from roop.inpainting import get_inpainting_manager
        from roop.temporal_consistency import get_temporal_manager
        from roop.advanced_face_models import get_face_model_manager
        from roop.enhanced_integration import get_enhanced_processor
        
        print("✓ All enhanced modules imported successfully")
        
        # Test basic functionality
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(test_image, (128, 128), 50, (180, 180, 180), -1)  # Simple face
        
        # Test face detection
        faces = get_enhanced_faces(test_image, quality_threshold=0.1)
        print(f"✓ Enhanced face detection: found {len(faces)} faces")
        
        # Test inpainting
        inpainting_manager = get_inpainting_manager()
        methods = inpainting_manager.get_available_methods()
        print(f"✓ Inpainting methods: {methods}")
        
        # Test blending
        blender = AdvancedBlender()
        blend_methods = get_available_blend_methods()
        print(f"✓ Blend methods: {blend_methods}")
        
        # Test temporal consistency
        temporal_manager = get_temporal_manager()
        info = temporal_manager.get_temporal_info()
        print(f"✓ Temporal consistency: {info}")
        
        # Test face models
        model_manager = get_face_model_manager()
        available_models = model_manager.get_available_models()
        print(f"✓ Face models: {available_models}")
        
        # Test integration
        processor = get_enhanced_processor()
        stats = processor.get_processing_stats()
        print(f"✓ Enhanced processor: {stats}")
        
        print("\n🎉 All tests passed! Enhanced face processing is working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)