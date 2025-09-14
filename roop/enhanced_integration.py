"""
Integration module for enhanced face processing features.
Provides unified interface to new capabilities while maintaining compatibility.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

from roop.typing import Frame, Face
from roop.inpainting import (
    get_inpainting_manager, 
    enhance_face_boundaries, 
    get_available_inpainting_methods
)
from roop.temporal_consistency import (
    get_temporal_manager,
    process_frame_with_temporal_consistency
)
from roop.advanced_face_models import (
    get_face_model_manager,
    enhance_face_with_wan,
    select_best_face
)
from roop.enhanced_face_detection import get_enhanced_faces
from roop.advanced_blending import AdvancedBlender, get_available_blend_methods

logger = logging.getLogger(__name__)


class EnhancedProcessingConfig:
    """Configuration for enhanced processing features."""
    
    def __init__(self):
        # Inpainting settings
        self.enable_inpainting = False
        self.inpainting_method = "traditional_telea"
        self.boundary_enhancement = True
        self.boundary_blend_strength = 0.7
        self.mask_expansion_ratio = 0.1
        self.inpaint_radius = 3
        self.inpaint_prompt = "natural face, high quality"
        
        # Temporal consistency settings
        self.enable_temporal_consistency = True
        self.temporal_smoothing_factor = 0.3
        self.enable_optical_flow = True
        self.enable_position_stabilization = True
        self.enable_landmark_stabilization = True
        self.quality_filter_threshold = 0.5
        self.temporal_buffer_size = 5
        
        # Advanced face model settings
        self.face_enhancement_model = "none"
        self.face_enhancement_level = 0.5
        self.enable_quality_analysis = True
        self.quality_enhancement_threshold = 0.3
        self.enable_best_face_selection = False
        
        # Optimization settings
        self.enable_memory_optimization = True
        self.adaptive_quality_settings = True
        self.parallel_face_processing = False
        self.batch_size_limit = 10
        self.memory_usage_limit = 8
        self.processing_priority = "balanced"


class EnhancedFaceProcessor:
    """Main processor that integrates all enhanced features."""
    
    def __init__(self, config: Optional[EnhancedProcessingConfig] = None):
        self.config = config or EnhancedProcessingConfig()
        
        # Initialize managers
        self.inpainting_manager = get_inpainting_manager()
        self.temporal_manager = get_temporal_manager()
        self.face_model_manager = get_face_model_manager()
        self.blender = AdvancedBlender()
        
        # Frame counter for temporal processing
        self.frame_counter = 0
        self.is_video_processing = False
        
        # Performance tracking
        self.processing_stats = {
            'total_frames': 0,
            'enhanced_frames': 0,
            'inpainted_frames': 0,
            'temporal_processed_frames': 0,
            'average_processing_time': 0.0
        }
    
    def start_video_processing(self):
        """Initialize for video processing."""
        self.is_video_processing = True
        self.frame_counter = 0
        
        # Configure temporal consistency manager
        self.temporal_manager.enable_position_stabilization = self.config.enable_position_stabilization
        self.temporal_manager.enable_landmark_stabilization = self.config.enable_landmark_stabilization
        self.temporal_manager.enable_optical_flow = self.config.enable_optical_flow
        self.temporal_manager.quality_threshold = self.config.quality_filter_threshold
        
        # Reset temporal state
        self.temporal_manager.reset()
        
        logger.info("Enhanced video processing started")
    
    def finish_video_processing(self):
        """Cleanup after video processing."""
        self.is_video_processing = False
        logger.info(f"Enhanced video processing finished. Stats: {self.processing_stats}")
    
    def process_frame(self, source_face: Face, target_frame: Frame, 
                     target_faces: Optional[List[Face]] = None) -> Tuple[Frame, Dict[str, Any]]:
        """
        Process a single frame with all enhancements.
        
        Args:
            source_face: Source face for swapping
            target_frame: Target frame to process
            target_faces: Optional pre-detected target faces
            
        Returns:
            Tuple of (processed_frame, processing_metadata)
        """
        start_time = cv2.getTickCount()
        
        metadata = {
            'frame_number': self.frame_counter,
            'enhancements_applied': [],
            'processing_time': 0.0,
            'quality_metrics': {},
            'temporal_info': {}
        }
        
        # Make a copy to work with
        result_frame = target_frame.copy()
        
        # Step 1: Detect faces if not provided
        if target_faces is None:
            target_faces = get_enhanced_faces(target_frame)
        
        if not target_faces:
            metadata['error'] = 'No faces detected'
            return result_frame, metadata
        
        # Step 2: Face quality analysis and selection
        if self.config.enable_quality_analysis:
            if self.config.enable_best_face_selection and len(target_faces) > 1:
                # Select best quality face
                face_candidates = [(face, self._extract_face_image(target_frame, face)) 
                                 for face in target_faces]
                best_face, best_image, selection_metadata = select_best_face(face_candidates)
                
                if best_face is not None:
                    target_faces = [best_face]
                    metadata['quality_metrics'] = selection_metadata
                    metadata['enhancements_applied'].append('best_face_selection')
        
        # Step 3: Temporal consistency (for video)
        if self.is_video_processing and self.config.enable_temporal_consistency:
            stabilized_faces, temporal_metadata = process_frame_with_temporal_consistency(
                target_frame, target_faces, self.frame_counter
            )
            target_faces = stabilized_faces
            metadata['temporal_info'] = temporal_metadata
            metadata['enhancements_applied'].append('temporal_consistency')
            self.processing_stats['temporal_processed_frames'] += 1
        
        # Step 4: Face enhancement (before swapping)
        enhanced_faces = []
        for face in target_faces:
            if self.config.face_enhancement_model != "none":
                face_image = self._extract_face_image(target_frame, face)
                
                if self.config.face_enhancement_model == "wan_enhancement":
                    enhanced_image, enhance_metadata = enhance_face_with_wan(
                        face_image, self.config.face_enhancement_level
                    )
                    
                    # Apply enhancement back to frame
                    if enhance_metadata.get('enhancement_applied', False):
                        result_frame = self._apply_face_image(result_frame, face, enhanced_image)
                        metadata['enhancements_applied'].append('wan_enhancement')
                        metadata['quality_metrics'].update(enhance_metadata)
            
            enhanced_faces.append(face)
        
        # Step 5: Face swapping (using existing enhanced face swapper)
        from roop.enhanced_face_swapper import enhanced_swap_face
        
        for face in enhanced_faces:
            try:
                result_frame = enhanced_swap_face(
                    source_face, face, result_frame,
                    blend_method=self._get_blend_method(),
                    blend_ratio=0.8
                )
                metadata['enhancements_applied'].append('face_swap')
            except Exception as e:
                logger.warning(f"Face swap failed: {e}")
                continue
        
        # Step 6: Post-processing inpainting
        if self.config.enable_inpainting and enhanced_faces:
            for face in enhanced_faces:
                try:
                    if self.config.boundary_enhancement:
                        result_frame = enhance_face_boundaries(
                            result_frame, target_frame, face,
                            self.config.boundary_blend_strength
                        )
                        metadata['enhancements_applied'].append('boundary_enhancement')
                    
                    # Additional inpainting if needed
                    if self.config.inpainting_method != "traditional_telea":
                        inpainted_frame, mask = self.inpainting_manager.inpaint_face_region(
                            result_frame, face, self.config.inpainting_method,
                            mask_type='boundary',
                            expand_ratio=self.config.mask_expansion_ratio,
                            inpaint_radius=self.config.inpaint_radius,
                            prompt=self.config.inpaint_prompt
                        )
                        result_frame = inpainted_frame
                        metadata['enhancements_applied'].append('inpainting')
                        self.processing_stats['inpainted_frames'] += 1
                        
                except Exception as e:
                    logger.warning(f"Inpainting failed: {e}")
                    continue
        
        # Update statistics
        processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        metadata['processing_time'] = processing_time
        
        self.processing_stats['total_frames'] += 1
        if len(metadata['enhancements_applied']) > 1:  # More than just face_swap
            self.processing_stats['enhanced_frames'] += 1
        
        # Update average processing time
        self.processing_stats['average_processing_time'] = (
            (self.processing_stats['average_processing_time'] * (self.processing_stats['total_frames'] - 1) + 
             processing_time) / self.processing_stats['total_frames']
        )
        
        self.frame_counter += 1
        
        return result_frame, metadata
    
    def _extract_face_image(self, frame: Frame, face: Face) -> np.ndarray:
        """Extract face image from frame."""
        if hasattr(face, 'bbox'):
            x1, y1, x2, y2 = map(int, face.bbox)
            return frame[y1:y2, x1:x2]
        return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def _apply_face_image(self, frame: Frame, face: Face, face_image: np.ndarray) -> Frame:
        """Apply face image back to frame."""
        if hasattr(face, 'bbox') and face_image.size > 0:
            x1, y1, x2, y2 = map(int, face.bbox)
            h, w = y2 - y1, x2 - x1
            
            if h > 0 and w > 0:
                face_resized = cv2.resize(face_image, (w, h))
                frame[y1:y2, x1:x2] = face_resized
        
        return frame
    
    def _get_blend_method(self) -> str:
        """Get appropriate blend method based on processing priority."""
        if self.config.processing_priority == "quality":
            return "multiband"
        elif self.config.processing_priority == "speed":
            return "alpha"
        else:  # balanced
            return "poisson"
    
    def get_available_methods(self) -> Dict[str, List[str]]:
        """Get available methods for all enhancement types."""
        return {
            'inpainting_methods': get_available_inpainting_methods(),
            'blend_methods': get_available_blend_methods(),
            'face_models': self.face_model_manager.get_available_models()
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")


# Global processor instance
ENHANCED_PROCESSOR = None


def get_enhanced_processor() -> EnhancedFaceProcessor:
    """Get global enhanced processor instance."""
    global ENHANCED_PROCESSOR
    if ENHANCED_PROCESSOR is None:
        ENHANCED_PROCESSOR = EnhancedFaceProcessor()
    return ENHANCED_PROCESSOR


def process_frame_enhanced(source_face: Face, target_frame: Frame, 
                         **config_updates) -> Tuple[Frame, Dict[str, Any]]:
    """Convenience function for enhanced frame processing."""
    processor = get_enhanced_processor()
    
    # Update configuration if provided
    if config_updates:
        processor.update_config(**config_updates)
    
    return processor.process_frame(source_face, target_frame)


def start_video_processing_enhanced(**config_updates):
    """Start enhanced video processing."""
    processor = get_enhanced_processor()
    
    if config_updates:
        processor.update_config(**config_updates)
    
    processor.start_video_processing()


def finish_video_processing_enhanced():
    """Finish enhanced video processing."""
    processor = get_enhanced_processor()
    processor.finish_video_processing()


def get_enhanced_processing_stats() -> Dict[str, Any]:
    """Get enhanced processing statistics."""
    return get_enhanced_processor().get_processing_stats()


def get_available_enhancement_methods() -> Dict[str, List[str]]:
    """Get all available enhancement methods."""
    return get_enhanced_processor().get_available_methods()