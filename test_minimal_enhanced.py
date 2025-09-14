#!/usr/bin/env python3
"""
Minimal test for enhanced modules without UI dependencies.
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/roop-unleashed/roop-unleashed')

def test_individual_modules():
    """Test each enhanced module individually."""
    print("=== Testing Enhanced Modules Individually ===")
    
    # Test 1: Inpainting module (standalone)
    try:
        from roop.inpainting import get_inpainting_manager, get_available_inpainting_methods
        manager = get_inpainting_manager()
        methods = get_available_inpainting_methods()
        print(f"✓ Inpainting module: {methods}")
    except Exception as e:
        print(f"✗ Inpainting module failed: {e}")
        return False
    
    # Test 2: Advanced blending (standalone)
    try:
        from roop.advanced_blending import AdvancedBlender, get_available_blend_methods
        blender = AdvancedBlender()
        methods = get_available_blend_methods()
        print(f"✓ Advanced blending: {methods}")
    except Exception as e:
        print(f"✗ Advanced blending failed: {e}")
        return False
    
    # Test 3: Temporal consistency (standalone)
    try:
        from roop.temporal_consistency import get_temporal_manager, get_temporal_info
        manager = get_temporal_manager()
        info = get_temporal_info()
        print(f"✓ Temporal consistency: buffer_size={info.get('buffer_size', 0)}")
    except Exception as e:
        print(f"✗ Temporal consistency failed: {e}")
        return False
    
    # Test 4: Advanced face models (standalone)
    try:
        from roop.advanced_face_models import get_face_model_manager, FaceQualityAnalyzer
        manager = get_face_model_manager()
        analyzer = FaceQualityAnalyzer()
        models = manager.get_available_models()
        print(f"✓ Advanced face models: {models}")
    except Exception as e:
        print(f"✗ Advanced face models failed: {e}")
        return False
    
    # Test 5: Enhanced face detection (needs mock insightface)
    try:
        from roop.enhanced_face_detection import get_enhanced_faces, FaceQualityAssessment
        test_image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(test_image, (64, 64), 30, (180, 180, 180), -1)
        
        faces = get_enhanced_faces(test_image, quality_threshold=0.1)
        print(f"✓ Enhanced face detection: found {len(faces)} faces")
    except Exception as e:
        print(f"✗ Enhanced face detection failed: {e}")
        return False
    
    # Test 6: Enhanced integration (avoid face swapper import)
    try:
        from roop.enhanced_integration import EnhancedProcessingConfig, get_available_enhancement_methods
        config = EnhancedProcessingConfig()
        methods = get_available_enhancement_methods()
        print(f"✓ Enhanced integration config: {len(methods)} method types")
    except Exception as e:
        print(f"✗ Enhanced integration failed: {e}")
        return False
    
    print("\n🎉 All individual enhanced modules working correctly!")
    return True

def test_functionality():
    """Test actual functionality of enhanced modules."""
    print("\n=== Testing Enhanced Module Functionality ===")
    
    try:
        # Test inpainting functionality
        from roop.inpainting import get_inpainting_manager
        manager = get_inpainting_manager()
        
        # Create test image and mask
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[50:78, 50:78] = 255
        
        # Test traditional inpainting
        inpainter = manager.inpainters['traditional_telea']
        result = inpainter.inpaint(test_image, mask, inpaint_radius=3)
        print(f"✓ Inpainting functionality: {result.shape}")
        
        # Test face quality analysis
        from roop.advanced_face_models import FaceQualityAnalyzer
        analyzer = FaceQualityAnalyzer()
        
        face_image = np.ones((64, 64, 3), dtype=np.uint8) * 128
        metrics = analyzer.analyze_face(face_image)
        overall_quality = analyzer.get_overall_quality(metrics)
        print(f"✓ Face quality analysis: score={overall_quality:.3f}")
        
        # Test blending
        from roop.advanced_blending import AdvancedBlender
        blender = AdvancedBlender()
        
        source = np.ones((64, 64, 3), dtype=np.uint8) * 200
        target_frame = np.ones((128, 128, 3), dtype=np.uint8) * 100
        bbox = (32, 32, 96, 96)
        
        result = blender.blend_face(source, target_frame, bbox, "alpha", 0.8)
        print(f"✓ Advanced blending: {result.shape}")
        
        print("\n🎉 Enhanced module functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_individual_modules()
    success2 = test_functionality() if success1 else False
    
    if success1 and success2:
        print("\n✅ All enhanced features are working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Some enhanced features need attention.")
        sys.exit(1)