"""
Enhanced face detection module with improved accuracy and quality assessment.
Implements adaptive detection, multi-scale analysis, and face quality scoring.
"""

import cv2
import numpy as np
import threading
from typing import List, Tuple, Optional, Any

import roop.globals
from roop.typing import Face, Frame

# Try to import insightface, fall back to mock if not available
try:
    import insightface
except ImportError:
    print("insightface not found, using mock implementation")
    from roop import mock_insightface as insightface

# Global variables for enhanced face detection
ENHANCED_FACE_ANALYSER = None
THREAD_LOCK_ENHANCED = threading.Lock()


class FaceQualityAssessment:
    """Assess face quality based on multiple criteria."""
    
    @staticmethod
    def calculate_face_quality(face: Face, frame: Frame) -> float:
        """
        Calculate face quality score based on multiple factors.
        Returns score between 0.0 and 1.0 (higher is better).
        """
        quality_score = 0.0
        
        # Detection confidence score (0.3 weight)
        detection_score = min(face.det_score, 1.0)
        quality_score += detection_score * 0.3
        
        # Face size score (0.2 weight)
        bbox = face.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = frame.shape[0] * frame.shape[1]
        size_ratio = face_area / frame_area
        size_score = min(size_ratio * 10, 1.0)  # Optimal around 10% of frame
        quality_score += size_score * 0.2
        
        # Pose estimation score (0.2 weight)
        pose_score = FaceQualityAssessment._calculate_pose_score(face)
        quality_score += pose_score * 0.2
        
        # Sharpness score (0.15 weight)
        sharpness_score = FaceQualityAssessment._calculate_sharpness(face, frame)
        quality_score += sharpness_score * 0.15
        
        # Lighting score (0.15 weight)
        lighting_score = FaceQualityAssessment._calculate_lighting_score(face, frame)
        quality_score += lighting_score * 0.15
        
        return min(quality_score, 1.0)
    
    @staticmethod
    def _calculate_pose_score(face: Face) -> float:
        """Calculate face pose score based on landmark positions."""
        if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
            return 0.5  # Default score if no landmarks
        
        landmarks = face.landmark_2d_106
        
        # Calculate eye distance to assess frontal pose
        left_eye = np.mean(landmarks[33:42], axis=0)
        right_eye = np.mean(landmarks[42:51], axis=0)
        nose_tip = landmarks[86]
        
        # Calculate symmetry
        eye_center = (left_eye + right_eye) / 2
        nose_to_eye_center = abs(nose_tip[0] - eye_center[0])
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        symmetry_ratio = 1.0 - min(nose_to_eye_center / (eye_distance / 2), 1.0)
        
        return max(symmetry_ratio, 0.0)
    
    @staticmethod
    def _calculate_sharpness(face: Face, frame: Frame) -> float:
        """Calculate face sharpness using Laplacian variance."""
        bbox = face.bbox.astype(int)
        face_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if face_region.size == 0:
            return 0.0
        
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Normalize sharpness score (empirically determined thresholds)
        return min(laplacian_var / 1000.0, 1.0)
    
    @staticmethod
    def _calculate_lighting_score(face: Face, frame: Frame) -> float:
        """Calculate lighting quality based on histogram distribution."""
        bbox = face.bbox.astype(int)
        face_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if face_region.size == 0:
            return 0.0
        
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        
        # Calculate distribution uniformity
        hist_norm = hist / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        # Normalize entropy score (max entropy for 8-bit is 8)
        return min(entropy / 8.0, 1.0)


class AdaptiveFaceDetection:
    """Enhanced face detection with adaptive parameters."""
    
    @staticmethod
    def get_optimal_detection_size(frame: Frame) -> Tuple[int, int]:
        """
        Calculate optimal detection size based on frame resolution.
        Returns (width, height) for detection.
        """
        height, width = frame.shape[:2]
        total_pixels = height * width
        
        # Adaptive sizing based on frame resolution
        if total_pixels > 2073600:  # 1920x1080 and above
            return (640, 640)
        elif total_pixels > 921600:  # 1280x720 and above
            return (512, 512)
        elif total_pixels > 307200:  # 640x480 and above
            return (416, 416)
        else:
            return (320, 320)
    
    @staticmethod
    def multi_scale_detection(frame: Frame, analyser: Any) -> List[Face]:
        """
        Perform multi-scale face detection for better accuracy.
        """
        faces_all_scales = []
        
        # Original scale
        faces = analyser.get(frame)
        if faces:
            faces_all_scales.extend(faces)
        
        # Scale up for small faces
        height, width = frame.shape[:2]
        if min(height, width) > 640:
            scale_factor = 1.5
            scaled_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            
            scaled_faces = analyser.get(scaled_frame)
            if scaled_faces:
                # Adjust bounding boxes back to original scale
                for face in scaled_faces:
                    face.bbox = face.bbox / scale_factor
                    if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                        face.landmark_2d_106 = face.landmark_2d_106 / scale_factor
                faces_all_scales.extend(scaled_faces)
        
        # Remove duplicate detections using NMS
        return AdaptiveFaceDetection._non_max_suppression(faces_all_scales)
    
    @staticmethod
    def _non_max_suppression(faces: List[Face], overlap_threshold: float = 0.3) -> List[Face]:
        """Apply non-maximum suppression to remove duplicate detections."""
        if not faces:
            return []
        
        # Sort faces by detection score
        faces_sorted = sorted(faces, key=lambda x: x.det_score, reverse=True)
        
        keep = []
        while faces_sorted:
            current = faces_sorted.pop(0)
            keep.append(current)
            
            # Remove overlapping faces
            faces_sorted = [
                face for face in faces_sorted
                if AdaptiveFaceDetection._calculate_iou(current.bbox, face.bbox) < overlap_threshold
            ]
        
        return keep
    
    @staticmethod
    def _calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def get_enhanced_face_analyser() -> Any:
    """Get enhanced face analyser with adaptive configuration."""
    global ENHANCED_FACE_ANALYSER
    
    with THREAD_LOCK_ENHANCED:
        if ENHANCED_FACE_ANALYSER is None:
            # Handle cases where CFG might not be initialized
            force_cpu = False
            if hasattr(roop.globals, 'CFG') and roop.globals.CFG is not None:
                force_cpu = getattr(roop.globals.CFG, 'force_cpu', False)
            
            if force_cpu:
                print('Forcing CPU for Enhanced Face Analysis')
                ENHANCED_FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name='buffalo_l', 
                    providers=['CPUExecutionProvider']
                )
            else:
                execution_providers = getattr(roop.globals, 'execution_providers', ['CPUExecutionProvider'])
                ENHANCED_FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name='buffalo_l', 
                    providers=execution_providers
                )
            
            # Use higher resolution by default for better accuracy
            ENHANCED_FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    
    return ENHANCED_FACE_ANALYSER


def get_enhanced_faces(frame: Frame, quality_threshold: float = 0.5) -> List[Tuple[Face, float]]:
    """
    Get faces with enhanced detection and quality assessment.
    Returns list of (face, quality_score) tuples sorted by quality.
    """
    analyser = get_enhanced_face_analyser()
    
    # Adaptive detection size
    optimal_size = AdaptiveFaceDetection.get_optimal_detection_size(frame)
    if optimal_size != (640, 640):
        analyser.prepare(ctx_id=0, det_size=optimal_size)
    
    # Multi-scale detection
    faces = AdaptiveFaceDetection.multi_scale_detection(frame, analyser)
    
    if not faces:
        return []
    
    # Calculate quality scores
    faces_with_quality = []
    for face in faces:
        quality_score = FaceQualityAssessment.calculate_face_quality(face, frame)
        if quality_score >= quality_threshold:
            faces_with_quality.append((face, quality_score))
    
    # Sort by quality score (best first)
    faces_with_quality.sort(key=lambda x: x[1], reverse=True)
    
    return faces_with_quality


def get_best_face(frame: Frame) -> Optional[Face]:
    """Get the highest quality face from the frame."""
    faces_with_quality = get_enhanced_faces(frame)
    return faces_with_quality[0][0] if faces_with_quality else None


def get_all_quality_faces(frame: Frame, min_quality: float = 0.3) -> List[Face]:
    """Get all faces above minimum quality threshold, sorted by quality."""
    faces_with_quality = get_enhanced_faces(frame, min_quality)
    return [face for face, _ in faces_with_quality]