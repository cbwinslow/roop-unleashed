"""
Enhanced face swapper with improved detection, quality assessment, and blending.
Integrates advanced detection and blending techniques for superior results.
"""

from typing import Any, List, Callable, Tuple, Optional
import cv2
import numpy as np
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video, compute_cosine_distance, get_destfilename_from_path

# Import our enhanced modules
from roop.enhanced_face_detection import get_enhanced_faces, get_best_face, FaceQualityAssessment
from roop.advanced_blending import AdvancedBlender, get_available_blend_methods

ENHANCED_FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.ENHANCED-FACE-SWAPPER'

# Enhanced parameters
QUALITY_THRESHOLD = 0.4
DISTANCE_THRESHOLD = 0.65
DEFAULT_BLEND_METHOD = "multiband"


class EnhancedFacePreprocessor:
    """Preprocessing for faces before swapping."""
    
    @staticmethod
    def normalize_face(face_image: np.ndarray) -> np.ndarray:
        """Normalize face image for better swapping results."""
        # Histogram equalization for better lighting
        if len(face_image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            face_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Gentle noise reduction
        face_image = cv2.bilateralFilter(face_image, 5, 50, 50)
        
        return face_image
    
    @staticmethod
    def enhance_face_alignment(source_face: Face, target_face: Face) -> Tuple[Face, Face]:
        """Improve face alignment for better swapping."""
        # This is a placeholder for advanced alignment techniques
        # In a full implementation, you would use face landmark normalization
        return source_face, target_face


class EnhancedSwapQuality:
    """Quality assessment and validation for face swaps."""
    
    @staticmethod
    def assess_swap_quality(original_face: Face, swapped_region: np.ndarray, 
                          target_frame: np.ndarray) -> float:
        """
        Assess the quality of a face swap result.
        Returns quality score between 0.0 and 1.0.
        """
        quality_score = 0.0
        
        # Check for artifacts (simplified)
        artifacts_score = EnhancedSwapQuality._detect_artifacts(swapped_region)
        quality_score += artifacts_score * 0.4
        
        # Check color consistency
        color_score = EnhancedSwapQuality._assess_color_consistency(
            swapped_region, target_frame, original_face.bbox
        )
        quality_score += color_score * 0.3
        
        # Check edge quality
        edge_score = EnhancedSwapQuality._assess_edge_quality(swapped_region)
        quality_score += edge_score * 0.3
        
        return min(quality_score, 1.0)
    
    @staticmethod
    def _detect_artifacts(face_region: np.ndarray) -> float:
        """Detect visual artifacts in swapped face."""
        # Simple artifact detection using gradient analysis
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradient variance might indicate artifacts
        gradient_variance = np.var(gradient_magnitude)
        
        # Normalize score (empirically determined threshold)
        artifact_score = max(0.0, 1.0 - gradient_variance / 10000.0)
        return artifact_score
    
    @staticmethod
    def _assess_color_consistency(face_region: np.ndarray, full_frame: np.ndarray,
                                face_bbox: np.ndarray) -> float:
        """Assess color consistency with surrounding areas."""
        x1, y1, x2, y2 = face_bbox.astype(int)
        
        # Get surrounding context
        margin = 20
        context_x1 = max(0, x1 - margin)
        context_y1 = max(0, y1 - margin)
        context_x2 = min(full_frame.shape[1], x2 + margin)
        context_y2 = min(full_frame.shape[0], y2 + margin)
        
        context_region = full_frame[context_y1:context_y2, context_x1:context_x2]
        
        # Compare color distributions
        face_hist = cv2.calcHist([face_region], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        context_hist = cv2.calcHist([context_region], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        
        # Calculate histogram correlation
        correlation = cv2.compareHist(face_hist, context_hist, cv2.HISTCMP_CORREL)
        return max(0.0, correlation)
    
    @staticmethod
    def _assess_edge_quality(face_region: np.ndarray) -> float:
        """Assess the quality of face edges."""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge continuity (simplified metric)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Optimal edge ratio (empirically determined)
        optimal_ratio = 0.1
        edge_score = 1.0 - abs(edge_ratio - optimal_ratio) / optimal_ratio
        
        return max(0.0, edge_score)


def get_enhanced_face_swapper() -> Any:
    """Get enhanced face swapper with optimized settings."""
    global ENHANCED_FACE_SWAPPER
    
    with THREAD_LOCK:
        if ENHANCED_FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            ENHANCED_FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, 
                providers=roop.globals.execution_providers
            )
    return ENHANCED_FACE_SWAPPER


def enhanced_swap_face(source_face: Face, target_face: Face, temp_frame: Frame,
                      blend_method: str = DEFAULT_BLEND_METHOD,
                      blend_ratio: float = 0.8) -> Frame:
    """Enhanced face swapping with preprocessing and advanced blending."""
    
    # Preprocess faces for better alignment
    preprocessor = EnhancedFacePreprocessor()
    source_face, target_face = preprocessor.enhance_face_alignment(source_face, target_face)
    
    # Get initial swap result from base model
    swapper = get_enhanced_face_swapper()
    initial_result = swapper.get(temp_frame, target_face, source_face, paste_back=False)
    
    # Extract face region for advanced blending
    bbox = target_face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    
    # Normalize the swapped face
    swapped_face = preprocessor.normalize_face(initial_result)
    
    # Apply advanced blending
    blender = AdvancedBlender()
    final_result = blender.blend_face(
        swapped_face, temp_frame, (x1, y1, x2, y2), 
        blend_method, blend_ratio
    )
    
    return final_result


def enhanced_process_frame(source_face: Face, target_face: Face, temp_frame: Frame,
                         face_selection_mode: str = "best_quality",
                         blend_method: str = DEFAULT_BLEND_METHOD,
                         blend_ratio: float = 0.8,
                         min_quality: float = QUALITY_THRESHOLD) -> Frame:
    """
    Enhanced frame processing with quality-based face selection.
    
    Args:
        source_face: Source face for swapping
        target_face: Target face (optional, for face matching)
        temp_frame: Frame to process
        face_selection_mode: "best_quality", "all_faces", "match_target"
        blend_method: Blending method to use
        blend_ratio: Blending strength
        min_quality: Minimum quality threshold for faces
    """
    
    # Get enhanced face detection results
    faces_with_quality = get_enhanced_faces(temp_frame, min_quality)
    
    if not faces_with_quality:
        return temp_frame
    
    result_frame = temp_frame.copy()
    
    if face_selection_mode == "best_quality":
        # Swap only the highest quality face
        best_face, quality = faces_with_quality[0]
        result_frame = enhanced_swap_face(
            source_face, best_face, result_frame, blend_method, blend_ratio
        )
        
    elif face_selection_mode == "all_faces":
        # Swap all faces above quality threshold
        for face, quality in faces_with_quality:
            result_frame = enhanced_swap_face(
                source_face, face, result_frame, blend_method, blend_ratio
            )
            
    elif face_selection_mode == "match_target" and target_face is not None:
        # Match faces based on embedding similarity
        target_embedding = target_face.embedding
        
        for face, quality in faces_with_quality:
            distance = compute_cosine_distance(target_embedding, face.embedding)
            if distance <= DISTANCE_THRESHOLD:
                result_frame = enhanced_swap_face(
                    source_face, face, result_frame, blend_method, blend_ratio
                )
                break
    
    return result_frame


def enhanced_process_frames(is_batch: bool, source_face: Face, target_face: Face, 
                          temp_frame_paths: List[str], update: Callable[[], None],
                          blend_method: str = DEFAULT_BLEND_METHOD,
                          blend_ratio: float = 0.8) -> None:
    """Enhanced batch frame processing."""
    
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is not None:
            # Determine face selection mode
            selection_mode = "all_faces" if roop.globals.many_faces else "match_target"
            
            result = enhanced_process_frame(
                source_face, target_face, temp_frame,
                face_selection_mode=selection_mode,
                blend_method=blend_method,
                blend_ratio=blend_ratio
            )
            
            if result is not None:
                if is_batch:
                    output_path = get_destfilename_from_path(
                        temp_frame_path, roop.globals.output_path, '_enhanced.png'
                    )
                    cv2.imwrite(output_path, result)
                else:
                    cv2.imwrite(temp_frame_path, result)
                    
        if update:
            update()


# Main processing functions for compatibility with existing system
def enhanced_process_image(source_face: Any, target_face: Any, target_path: str, 
                         output_path: str, blend_method: str = DEFAULT_BLEND_METHOD) -> None:
    """Enhanced image processing."""
    target_frame = cv2.imread(target_path)
    if target_frame is not None:
        result = enhanced_process_frame(
            source_face, target_face, target_frame,
            blend_method=blend_method
        )
        if result is not None:
            cv2.imwrite(output_path, result)


def enhanced_process_video(source_face: Any, target_face: Any, temp_frame_paths: List[str],
                         blend_method: str = DEFAULT_BLEND_METHOD) -> None:
    """Enhanced video processing."""
    roop.processors.frame.core.process_video(
        source_face, target_face, temp_frame_paths, 
        lambda is_batch, src, tgt, paths, update: enhanced_process_frames(
            is_batch, src, tgt, paths, update, blend_method
        )
    )


def enhanced_process_batch_images(source_face: Any, target_face: Any, 
                                temp_frame_paths: List[str],
                                blend_method: str = DEFAULT_BLEND_METHOD) -> None:
    """Enhanced batch image processing."""
    roop.processors.frame.core.process_batch(
        source_face, target_face, temp_frame_paths,
        lambda is_batch, src, tgt, paths, update: enhanced_process_frames(
            is_batch, src, tgt, paths, update, blend_method
        )
    )


# Quality assessment functions
def assess_frame_quality(frame: Frame) -> dict:
    """Assess overall frame quality for face swapping."""
    faces_with_quality = get_enhanced_faces(frame, quality_threshold=0.0)
    
    if not faces_with_quality:
        return {
            "overall_quality": 0.0,
            "face_count": 0,
            "best_face_quality": 0.0,
            "average_face_quality": 0.0
        }
    
    qualities = [quality for _, quality in faces_with_quality]
    
    return {
        "overall_quality": max(qualities),
        "face_count": len(faces_with_quality),
        "best_face_quality": max(qualities),
        "average_face_quality": sum(qualities) / len(qualities)
    }


# Configuration and utility functions
def get_enhancement_config() -> dict:
    """Get current enhancement configuration."""
    return {
        "quality_threshold": QUALITY_THRESHOLD,
        "distance_threshold": DISTANCE_THRESHOLD,
        "default_blend_method": DEFAULT_BLEND_METHOD,
        "available_blend_methods": get_available_blend_methods()
    }


def set_enhancement_parameters(quality_threshold: float = None,
                             distance_threshold: float = None,
                             default_blend_method: str = None) -> None:
    """Set enhancement parameters."""
    global QUALITY_THRESHOLD, DISTANCE_THRESHOLD, DEFAULT_BLEND_METHOD
    
    if quality_threshold is not None:
        QUALITY_THRESHOLD = max(0.0, min(1.0, quality_threshold))
    
    if distance_threshold is not None:
        DISTANCE_THRESHOLD = max(0.0, min(2.0, distance_threshold))
    
    if default_blend_method is not None and default_blend_method in get_available_blend_methods():
        DEFAULT_BLEND_METHOD = default_blend_method