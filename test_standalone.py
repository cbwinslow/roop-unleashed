#!/usr/bin/env python3
"""
Standalone test for enhanced features - tests only our new modules without dependencies.
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/roop-unleashed/roop-unleashed')

def test_inpainting_module():
    """Test inpainting module functionality."""
    print("=== Testing Inpainting Module ===")
    
    try:
        # Import only the standalone parts
        from roop.inpainting import (
            InpaintingMaskGenerator, 
            TraditionalInpainter,
            InpaintingManager
        )
        
        # Test mask generation
        class MockFace:
            def __init__(self, bbox):
                self.bbox = bbox
        
        face = MockFace([50, 50, 150, 150])
        mask = InpaintingMaskGenerator.create_face_boundary_mask(
            face, (200, 200, 3), 0.1
        )
        
        print(f"✓ Mask generation successful. Mask shape: {mask.shape}")
        
        # Test traditional inpainting
        inpainter = TraditionalInpainter()
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = inpainter.inpaint(test_image, mask)
        
        print(f"✓ Traditional inpainting successful. Result shape: {result.shape}")
        
        # Test manager
        manager = InpaintingManager()
        available_methods = manager.get_available_methods()
        print(f"✓ Inpainting manager works. Available methods: {available_methods}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inpainting test failed: {e}")
        return False

def test_temporal_module():
    """Test temporal consistency module."""
    print("\n=== Testing Temporal Consistency Module ===")
    
    try:
        from roop.temporal_consistency import (
            TemporalBuffer,
            TemporalStabilizer,
            FrameInterpolator,
            FrameInfo
        )
        
        # Test temporal buffer
        buffer = TemporalBuffer(buffer_size=3)
        
        # Create mock frame info
        test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        frame_info = FrameInfo(
            frame_number=0,
            frame=test_frame,
            faces=[],
            face_embeddings=[],
            quality_scores=[0.7],
            processing_time=0.01
        )
        
        buffer.add_frame(frame_info)
        recent_frames = buffer.get_recent_frames(1)
        
        print(f"✓ Temporal buffer works. Buffer size: {len(buffer.frames)}")
        
        # Test stabilizer
        stabilizer = TemporalStabilizer()
        
        class MockFace:
            def __init__(self, bbox):
                self.bbox = bbox
        
        face = MockFace([50, 50, 150, 150])
        stabilized_face = stabilizer.stabilize_face_position(face, 0)
        
        print(f"✓ Temporal stabilizer works")
        
        # Test interpolator
        interpolator = FrameInterpolator()
        face1 = MockFace([50, 50, 150, 150])
        face2 = MockFace([60, 60, 160, 160])
        
        interpolated_face = interpolator.interpolate_face_features(face1, face2, 0.5)
        
        print(f"✓ Frame interpolator works")
        
        return True
        
    except Exception as e:
        print(f"✗ Temporal consistency test failed: {e}")
        return False

def test_face_models_module():
    """Test advanced face models module."""
    print("\n=== Testing Advanced Face Models Module ===")
    
    try:
        from roop.advanced_face_models import (
            FaceQualityAnalyzer,
            WANFaceModel,
            AdvancedFaceModelManager
        )
        
        # Test quality analyzer
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
        print(f"  Quality metrics: {list(quality_metrics.keys())}")
        print(f"  Overall quality: {overall_quality:.3f}")
        
        # Test WAN model (fallback mode)
        wan_model = WANFaceModel()
        enhanced_image = wan_model.enhance_face_quality(test_image, 0.5)
        
        print(f"✓ WAN model enhancement works (fallback mode)")
        
        # Test manager
        manager = AdvancedFaceModelManager()
        available_models = manager.get_available_models()
        
        print(f"✓ Face model manager works. Available models: {available_models}")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced face models test failed: {e}")
        return False

def test_integration_module():
    """Test enhanced integration module."""
    print("\n=== Testing Enhanced Integration Module ===")
    
    try:
        from roop.enhanced_integration import (
            EnhancedProcessingConfig,
            EnhancedFaceProcessor
        )
        
        # Test configuration
        config = EnhancedProcessingConfig()
        config.enable_inpainting = True
        config.face_enhancement_level = 0.7
        
        print(f"✓ Enhanced processing config created")
        print(f"  Inpainting enabled: {config.enable_inpainting}")
        print(f"  Enhancement level: {config.face_enhancement_level}")
        
        # Test processor creation (without full processing)
        processor = EnhancedFaceProcessor(config)
        
        # Test configuration updates
        processor.update_config(temporal_smoothing_factor=0.5)
        
        # Test statistics
        stats = processor.get_processing_stats()
        
        print(f"✓ Enhanced processor created and configured")
        print(f"  Initial stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced integration test failed: {e}")
        return False

def test_ui_integration():
    """Test that UI components are properly added."""
    print("\n=== Testing UI Integration ===")
    
    try:
        # Check if our UI functions are defined
        from roop.ui import (
            on_test_inpainting,
            on_reset_temporal,
            on_analyze_face_quality,
            get_system_info,
            get_temporal_info
        )
        
        print("✓ UI callback functions are defined")
        
        # Test system info function
        sys_info = get_system_info()
        print(f"✓ System info function works: {len(sys_info)} chars")
        
        return True
        
    except Exception as e:
        print(f"✗ UI integration test failed: {e}")
        return False

def main():
    """Run all standalone tests."""
    print("Enhanced Face Processing - Standalone Test Suite")
    print("=================================================")
    
    tests = [
        test_inpainting_module,
        test_temporal_module,
        test_face_models_module,
        test_integration_module,
        test_ui_integration
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
        print("🎉 All standalone tests passed! Enhanced modules are working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())