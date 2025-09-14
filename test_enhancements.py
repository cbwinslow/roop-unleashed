#!/usr/bin/env python3
"""
Comprehensive test script for enhanced face processing capabilities.
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/roop-unleashed/roop-unleashed')

def test_basic_imports():
    """Test basic module imports."""
    print("=== Testing Basic Imports ===")
    try:
        import roop.globals
        from roop.enhanced_face_detection import get_enhanced_faces, FaceQualityAssessment
        from roop.advanced_blending import AdvancedBlender, get_available_blend_methods
        from roop.enhanced_face_swapper import assess_frame_quality, get_enhancement_config
        print("✓ Basic enhanced modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Basic import error: {e}")
        return False

def test_new_modules():
    """Test new advanced modules."""
    print("\n=== Testing New Advanced Modules ===")
    
    try:
        # Test inpainting module
        from roop.inpainting import (
            get_inpainting_manager, 
            get_available_inpainting_methods,
            InpaintingMaskGenerator
        )
        
        manager = get_inpainting_manager()
        methods = get_available_inpainting_methods()
        print(f"✓ Inpainting module loaded. Available methods: {methods}")
        
        # Test temporal consistency
        from roop.temporal_consistency import (
            get_temporal_manager,
            get_temporal_info,
            reset_temporal_state
        )
        
        temporal_manager = get_temporal_manager()
        info = get_temporal_info()
        print(f"✓ Temporal consistency module loaded. Info: {info}")
        
        # Test advanced face models
        from roop.advanced_face_models import (
            get_face_model_manager,
            analyze_face_quality,
            FaceQualityAnalyzer
        )
        
        model_manager = get_face_model_manager()
        analyzer = FaceQualityAnalyzer()
        print(f"✓ Advanced face models loaded. Available models: {model_manager.get_available_models()}")
        
        # Test integration module
        from roop.enhanced_integration import (
            get_enhanced_processor,
            EnhancedProcessingConfig,
            get_available_enhancement_methods
        )
        
        processor = get_enhanced_processor()
        config = EnhancedProcessingConfig()
        methods = get_available_enhancement_methods()
        print(f"✓ Enhanced integration module loaded. Available methods: {list(methods.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"✗ New modules import error: {e}")
        return False
    except Exception as e:
        print(f"✗ New modules test error: {e}")
        return False

def test_face_quality_analysis():
    """Test face quality analysis functionality."""
    print("\n=== Testing Face Quality Analysis ===")
    
    try:
        from roop.advanced_face_models import FaceQualityAnalyzer
        
        analyzer = FaceQualityAnalyzer()
        
        # Create a test face image
        test_image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Draw a simple face
        center = (64, 64)
        cv2.circle(test_image, center, 40, (180, 180, 180), -1)  # Face
        cv2.circle(test_image, (50, 55), 4, (0, 0, 0), -1)       # Left eye
        cv2.circle(test_image, (78, 55), 4, (0, 0, 0), -1)       # Right eye
        cv2.ellipse(test_image, (64, 75), (10, 5), 0, 0, 180, (0, 0, 0), 1)  # Mouth
        
        # Analyze quality
        quality_metrics = analyzer.analyze_face(test_image)
        overall_quality = analyzer.get_overall_quality(quality_metrics)
        
        print(f"✓ Face quality analysis successful")
        print(f"  Quality metrics: {quality_metrics}")
        print(f"  Overall quality: {overall_quality:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Face quality analysis failed: {e}")
        return False

def test_inpainting():
    """Test inpainting functionality."""
    print("\n=== Testing Inpainting ===")
    
    try:
        from roop.inpainting import get_inpainting_manager, InpaintingMaskGenerator
        
        manager = get_inpainting_manager()
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Create a simple mask (rectangle in center)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:156, 100:156] = 255
        
        # Test traditional inpainting
        inpainter = manager.inpainters['traditional_telea']
        result = inpainter.inpaint(test_image, mask, inpaint_radius=3)
        
        print(f"✓ Traditional inpainting test successful")
        print(f"  Input shape: {test_image.shape}, Output shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inpainting test failed: {e}")
        return False

def test_temporal_consistency():
    """Test temporal consistency functionality."""
    print("\n=== Testing Temporal Consistency ===")
    
    try:
        from roop.temporal_consistency import get_temporal_manager, TemporalBuffer, FrameInfo
        
        manager = get_temporal_manager()
        buffer = TemporalBuffer(buffer_size=3)
        
        # Create mock frame info
        test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        frame_info = FrameInfo(
            frame_number=0,
            frame=test_frame,
            faces=[],
            face_embeddings=[],
            quality_scores=[],
            processing_time=0.01
        )
        
        buffer.add_frame(frame_info)
        recent_frames = buffer.get_recent_frames(1)
        
        print(f"✓ Temporal consistency test successful")
        print(f"  Buffer size: {len(buffer.frames)}")
        print(f"  Recent frames: {len(recent_frames)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Temporal consistency test failed: {e}")
        return False

def test_integration():
    """Test integration functionality."""
    print("\n=== Testing Enhanced Integration ===")
    
    try:
        from roop.enhanced_integration import (
            get_enhanced_processor,
            EnhancedProcessingConfig
        )
        
        # Create processor with custom config
        config = EnhancedProcessingConfig()
        config.enable_inpainting = True
        config.enable_temporal_consistency = False  # Disable for single frame test
        config.enable_quality_analysis = True
        
        processor = get_enhanced_processor()
        processor.config = config
        
        # Test configuration updates
        processor.update_config(face_enhancement_level=0.7)
        
        stats = processor.get_processing_stats()
        methods = processor.get_available_methods()
        
        print(f"✓ Enhanced integration test successful")
        print(f"  Processing stats: {stats}")
        print(f"  Available methods: {list(methods.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Enhanced Face Processing Test Suite")
    print("===================================")
    
    tests = [
        test_basic_imports,
        test_new_modules,
        test_face_quality_analysis,
        test_inpainting,
        test_temporal_consistency,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced face processing is ready to use!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())