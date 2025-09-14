"""
Integration module for enhanced face processing features.
Provides unified interface to new capabilities while maintaining compatibility.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import gc

from roop.typing import Frame, Face
from roop.enhanced_config import get_enhanced_config, EnhancedProcessingConfig
from roop.inpainting import get_inpainting_manager
from roop.temporal_consistency import get_temporal_manager
from roop.advanced_face_models import get_face_model_manager
from roop.video_frame_interpolation import get_frame_rate_enhancer
from roop.enhanced_face_profiler import get_enhanced_face_profiler
from roop.enhanced_face_detection import get_enhanced_faces
from roop.advanced_blending import AdvancedBlender, get_available_blend_methods

logger = logging.getLogger(__name__)


class EnhancedFaceProcessor:
    """
    Main processor that integrates all enhanced face processing features.
    Provides a unified interface for high-quality face swapping and enhancement.
    """
    
    def __init__(self, config: Optional[EnhancedProcessingConfig] = None):
        self.config = config or get_enhanced_config()
        
        # Initialize managers
        self.inpainting_manager = None
        self.temporal_manager = None
        self.face_model_manager = None
        self.frame_rate_enhancer = None
        self.face_profiler = None
        self.blender = None
        
        # Processing state
        self.is_video_processing = False
        self.frame_count = 0
        self.processing_stats = {}
        
        # Initialize components based on configuration
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize processing components based on configuration."""
        try:
            # Initialize inpainting if enabled
            if self.config.enable_enhanced_features:
                self.inpainting_manager = get_inpainting_manager()
                logger.info("Inpainting manager initialized")
            
            # Initialize temporal consistency if enabled
            if self.config.video.enable_temporal_consistency:
                self.temporal_manager = get_temporal_manager()
                logger.info("Temporal manager initialized")
            
            # Initialize face models
            self.face_model_manager = get_face_model_manager()
            logger.info("Face model manager initialized")
            
            # Initialize frame rate enhancer if enabled
            if self.config.video.enable_frame_interpolation:
                self.frame_rate_enhancer = get_frame_rate_enhancer(
                    self.config.video.target_fps,
                    self.config.video.interpolation_method.value
                )
                logger.info("Frame rate enhancer initialized")
            
            # Initialize face profiler
            self.face_profiler = get_enhanced_face_profiler()
            logger.info("Enhanced face profiler initialized")
            
            # Initialize blender
            self.blender = AdvancedBlender()
            logger.info("Advanced blender initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            # Fall back to basic functionality
            self.config.enable_enhanced_features = False
    
    def start_video_processing(self):
        """Initialize for video processing."""
        self.is_video_processing = True
        self.frame_count = 0
        self.processing_stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "faces_enhanced": 0,
            "obstructions_corrected": 0,
            "interpolated_frames": 0
        }
        
        # Reset temporal state
        if self.temporal_manager:
            self.temporal_manager.reset_state()
        
        logger.info("Video processing initialized")
    
    def finish_video_processing(self):
        """Cleanup after video processing."""
        self.is_video_processing = False
        
        # Log statistics
        if self.processing_stats:
            logger.info(f"Video processing completed: {self.processing_stats}")
        
        # Cleanup memory
        if self.config.performance.enable_memory_optimization:
            self._cleanup_memory()
    
    def process_frame(self, source_face: Face, target_frame: Frame, 
                     target_faces: Optional[List[Face]] = None) -> Tuple[Frame, Dict[str, Any]]:
        """
        Process a single frame with enhanced features.
        
        Args:
            source_face: Source face for swapping
            target_frame: Target frame to process
            target_faces: Optional list of target faces (will detect if None)
            
        Returns:
            Tuple of (processed_frame, processing_metadata)
        """
        metadata = {
            "frame_number": self.frame_count,
            "faces_detected": 0,
            "faces_processed": 0,
            "enhancements_applied": [],
            "processing_time": 0.0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Step 1: Face detection and profiling
            if target_faces is None:
                target_faces = get_enhanced_faces(target_frame)
            
            metadata["faces_detected"] = len(target_faces)
            
            if not target_faces:
                logger.debug("No faces detected in frame")
                return target_frame, metadata
            
            # Step 2: Face profiling and selection
            if self.config.face_profile.enable_quality_analysis:
                face_profiles = self.face_profiler.analyze_faces_in_frame(target_frame, target_faces)
                best_faces = self.face_profiler.select_best_faces(
                    face_profiles, 
                    max_faces=1 if not self.config.face_profile.enable_multi_angle_profiling else 3
                )
                target_faces = [profile.face for profile in best_faces]
                metadata["enhancements_applied"].append("face_profiling")
            
            if not target_faces:
                logger.debug("No suitable faces found after profiling")
                return target_frame, metadata
            
            # Step 3: Face enhancement (pre-processing)
            enhanced_source_face = self._enhance_source_face(source_face)
            
            # Step 4: Face swapping with advanced blending
            processed_frame = self._perform_face_swapping(
                enhanced_source_face, target_frame, target_faces[0]
            )
            metadata["faces_processed"] = 1
            
            # Step 5: Obstruction correction
            if self.config.inpainting.enable_smart_detection and self.inpainting_manager:
                corrected_frame, obstruction_mask = self.inpainting_manager.detect_and_correct_obstructions(
                    processed_frame, target_faces[0], self.config.inpainting.primary_method.value
                )
                if np.sum(obstruction_mask) > 0:
                    processed_frame = corrected_frame
                    metadata["enhancements_applied"].append("obstruction_correction")
                    self.processing_stats["obstructions_corrected"] += 1
            
            # Step 6: Temporal consistency (for video)
            if self.is_video_processing and self.config.video.enable_temporal_consistency and self.temporal_manager:
                processed_frame = self.temporal_manager.process_frame(
                    processed_frame, target_faces[0], self.config.video.temporal_smoothing_factor
                )
                metadata["enhancements_applied"].append("temporal_consistency")
            
            # Update statistics
            self.frame_count += 1
            self.processing_stats["frames_processed"] += 1
            self.processing_stats["faces_detected"] += metadata["faces_detected"]
            self.processing_stats["faces_enhanced"] += metadata["faces_processed"]
            
            # Calculate processing time
            metadata["processing_time"] = time.time() - start_time
            
            return processed_frame, metadata
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {e}")
            return target_frame, metadata
    
    def _enhance_source_face(self, source_face: Face) -> Face:
        """Enhance source face before swapping."""
        if not self.config.face_enhancement.enable_preprocessing:
            return source_face
        
        try:
            # Extract face image if needed
            if hasattr(source_face, 'image'):
                face_image = source_face.image
            else:
                # Would need to extract from original frame - placeholder
                return source_face
            
            # Apply enhancement based on configured model
            if self.config.face_enhancement.primary_model.value == "real_esrgan":
                # Use Real-ESRGAN enhancement
                enhanced_image = self.face_model_manager.enhance_with_realesrgan(
                    face_image, 
                    scale=self.config.face_enhancement.upscale_factor
                )
            elif self.config.face_enhancement.primary_model.value == "restore_former":
                # Use RestoreFormer enhancement
                enhanced_image = self.face_model_manager.enhance_with_restoreformer(
                    face_image,
                    restoration_level=self.config.face_enhancement.restoreformer_restoration_level
                )
            else:
                # Use WAN enhancement as fallback
                enhanced_image = self.face_model_manager.enhance_with_wan(
                    face_image,
                    enhancement_level=self.config.face_enhancement.enhancement_level
                )
            
            # Update face object with enhanced image
            if hasattr(source_face, 'image'):
                source_face.image = enhanced_image
            
            return source_face
            
        except Exception as e:
            logger.warning(f"Face enhancement failed: {e}")
            return source_face
    
    def _perform_face_swapping(self, source_face: Face, target_frame: Frame, target_face: Face) -> Frame:
        """Perform face swapping with advanced blending."""
        try:
            # Determine blending method based on performance settings
            if self.config.performance.priority.value == "quality":
                blend_method = "multiband"
            elif self.config.performance.priority.value == "speed":
                blend_method = "alpha"
            else:
                blend_method = "poisson"
            
            # Perform swapping with advanced blending
            swapped_frame = self.blender.blend_faces(
                source_face, target_face, target_frame, method=blend_method
            )
            
            return swapped_frame
            
        except Exception as e:
            logger.warning(f"Advanced face swapping failed: {e}")
            # Fallback to basic blending
            return self.blender.blend_faces(
                source_face, target_face, target_frame, method="alpha"
            )
    
    def enhance_video_frame_rate(self, frames: List[Frame], original_fps: float) -> Tuple[List[Frame], float]:
        """
        Enhance video frame rate using interpolation.
        
        Args:
            frames: List of video frames
            original_fps: Original frame rate
            
        Returns:
            Tuple of (enhanced_frames, new_fps)
        """
        if not self.config.video.enable_frame_interpolation or not self.frame_rate_enhancer:
            return frames, original_fps
        
        try:
            enhanced_frames, new_fps = self.frame_rate_enhancer.enhance_frame_rate(frames, original_fps)
            
            interpolated_count = len(enhanced_frames) - len(frames)
            self.processing_stats["interpolated_frames"] += interpolated_count
            
            logger.info(f"Enhanced frame rate from {original_fps} to {new_fps} FPS")
            logger.info(f"Added {interpolated_count} interpolated frames")
            
            return enhanced_frames, new_fps
            
        except Exception as e:
            logger.error(f"Frame rate enhancement failed: {e}")
            return frames, original_fps
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        stats["config"] = {
            "enhanced_features_enabled": self.config.enable_enhanced_features,
            "video_interpolation_enabled": self.config.video.enable_frame_interpolation,
            "temporal_consistency_enabled": self.config.video.enable_temporal_consistency,
            "smart_inpainting_enabled": self.config.inpainting.enable_smart_detection,
            "face_profiling_enabled": self.config.face_profile.enable_quality_analysis,
            "performance_priority": self.config.performance.priority.value
        }
        return stats
    
    def _cleanup_memory(self):
        """Cleanup memory and GPU resources."""
        try:
            # Force garbage collection
            if self.config.performance.enable_garbage_collection:
                gc.collect()
            
            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.debug("Memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")


class EnhancedVideoProcessor:
    """
    Video-specific processor that handles batch processing and optimization.
    """
    
    def __init__(self, config: Optional[EnhancedProcessingConfig] = None):
        self.config = config or get_enhanced_config()
        self.frame_processor = EnhancedFaceProcessor(config)
    
    def process_video_frames(self, source_face: Face, target_frames: List[Frame], 
                           original_fps: float) -> Tuple[List[Frame], float, Dict[str, Any]]:
        """
        Process a complete video with enhanced features.
        
        Args:
            source_face: Source face for swapping
            target_frames: List of target frames
            original_fps: Original video frame rate
            
        Returns:
            Tuple of (processed_frames, final_fps, processing_metadata)
        """
        self.frame_processor.start_video_processing()
        
        try:
            processed_frames = []
            
            # Process each frame
            for i, frame in enumerate(target_frames):
                processed_frame, frame_metadata = self.frame_processor.process_frame(
                    source_face, frame
                )
                processed_frames.append(processed_frame)
                
                if i % 10 == 0:  # Log progress every 10 frames
                    logger.info(f"Processed frame {i+1}/{len(target_frames)}")
            
            # Enhance frame rate if enabled
            if self.config.video.enable_frame_interpolation:
                processed_frames, final_fps = self.frame_processor.enhance_video_frame_rate(
                    processed_frames, original_fps
                )
            else:
                final_fps = original_fps
            
            # Get final statistics
            processing_metadata = self.frame_processor.get_processing_stats()
            
            return processed_frames, final_fps, processing_metadata
            
        finally:
            self.frame_processor.finish_video_processing()


# Global enhanced processor instance
_enhanced_processor = None


def get_enhanced_processor(config: Optional[EnhancedProcessingConfig] = None) -> EnhancedFaceProcessor:
    """Get the global enhanced face processor instance."""
    global _enhanced_processor
    if _enhanced_processor is None or config is not None:
        _enhanced_processor = EnhancedFaceProcessor(config)
    return _enhanced_processor


def process_frame_enhanced(source_face: Face, target_frame: Frame, 
                         config: Optional[EnhancedProcessingConfig] = None) -> Tuple[Frame, Dict[str, Any]]:
    """
    Convenience function for enhanced frame processing.
    
    Args:
        source_face: Source face for swapping
        target_frame: Target frame to process
        config: Optional configuration (uses global if None)
        
    Returns:
        Tuple of (processed_frame, processing_metadata)
    """
    processor = get_enhanced_processor(config)
    return processor.process_frame(source_face, target_frame)


def process_video_enhanced(source_face: Face, target_frames: List[Frame], 
                         original_fps: float,
                         config: Optional[EnhancedProcessingConfig] = None) -> Tuple[List[Frame], float, Dict[str, Any]]:
    """
    Convenience function for enhanced video processing.
    
    Args:
        source_face: Source face for swapping
        target_frames: List of target frames
        original_fps: Original video frame rate
        config: Optional configuration (uses global if None)
        
    Returns:
        Tuple of (processed_frames, final_fps, processing_metadata)
    """
    video_processor = EnhancedVideoProcessor(config)
    return video_processor.process_video_frames(source_face, target_frames, original_fps)
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