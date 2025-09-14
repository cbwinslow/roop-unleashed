"""
Enhanced Face Profile Builder with advanced detection and quality assessment.
Provides comprehensive face analysis, pose estimation, and profile generation.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
import math

from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path

logger = logging.getLogger(__name__)


class FaceQualityMetrics:
    """Comprehensive face quality assessment metrics."""
    
    @staticmethod
    def calculate_sharpness(face_image: np.ndarray) -> float:
        """Calculate face sharpness using Laplacian variance."""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range
        return min(laplacian_var / 1000.0, 1.0)
    
    @staticmethod
    def calculate_lighting_quality(face_image: np.ndarray) -> float:
        """Calculate lighting quality based on histogram distribution."""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate histogram spread (good lighting has wide distribution)
        hist_norm = hist.flatten() / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        # Normalize entropy (max is log2(256) = 8)
        return entropy / 8.0
    
    @staticmethod
    def calculate_resolution_quality(face_image: np.ndarray) -> float:
        """Calculate resolution quality based on face size."""
        h, w = face_image.shape[:2]
        face_area = h * w
        
        # Optimal face size is around 256x256 or larger
        optimal_area = 256 * 256
        quality = min(face_area / optimal_area, 1.0)
        
        return quality
    
    @staticmethod
    def calculate_pose_quality(landmarks: Optional[np.ndarray]) -> float:
        """Calculate pose quality (frontality) from facial landmarks."""
        if landmarks is None or len(landmarks) < 68:
            return 0.5  # Default moderate quality
        
        # Use key landmarks for pose estimation
        # Nose tip, left eye center, right eye center, mouth center
        nose_tip = landmarks[30]  # Nose tip
        left_eye = np.mean(landmarks[36:42], axis=0)  # Left eye
        right_eye = np.mean(landmarks[42:48], axis=0)  # Right eye
        mouth_center = np.mean(landmarks[48:68], axis=0)  # Mouth
        
        # Calculate symmetry
        eye_distance = np.linalg.norm(right_eye - left_eye)
        nose_to_left_eye = np.linalg.norm(nose_tip - left_eye)
        nose_to_right_eye = np.linalg.norm(nose_tip - right_eye)
        
        # Symmetry score (1.0 is perfect symmetry)
        if eye_distance > 0:
            symmetry = 1.0 - abs(nose_to_left_eye - nose_to_right_eye) / eye_distance
            symmetry = max(0.0, min(1.0, symmetry))
        else:
            symmetry = 0.5
        
        # Calculate head pose angle
        eye_center = (left_eye + right_eye) / 2
        face_angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        angle_score = 1.0 - abs(face_angle) / (math.pi / 4)  # Penalize angles > 45 degrees
        angle_score = max(0.0, min(1.0, angle_score))
        
        # Combine symmetry and angle
        pose_quality = (symmetry + angle_score) / 2.0
        
        return pose_quality
    
    @staticmethod
    def detect_artifacts(face_image: np.ndarray) -> float:
        """Detect compression artifacts and noise."""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Detect blocking artifacts using DCT
        dct_blocks = []
        h, w = gray.shape
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                dct_blocks.append(dct_block)
        
        if dct_blocks:
            # Calculate blocking artifact measure
            dct_array = np.array(dct_blocks)
            high_freq_energy = np.mean(np.abs(dct_array[:, 4:, 4:]))
            total_energy = np.mean(np.abs(dct_array))
            
            if total_energy > 0:
                artifact_ratio = high_freq_energy / total_energy
                quality = 1.0 - min(artifact_ratio * 10, 1.0)  # Scale and invert
            else:
                quality = 1.0
        else:
            quality = 1.0
        
        return quality


class PoseEstimator:
    """3D pose estimation for faces."""
    
    def __init__(self):
        # 3D model points for a generic face
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)
    
    def estimate_pose(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Estimate 3D pose from 2D facial landmarks.
        
        Args:
            landmarks: 2D facial landmarks
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary containing pose information
        """
        if len(landmarks) < 68:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.0}
        
        # Extract 2D image points from landmarks
        image_points = np.array([
            landmarks[30],    # Nose tip
            landmarks[8],     # Chin
            landmarks[36],    # Left eye left corner
            landmarks[45],    # Right eye right corner
            landmarks[48],    # Left mouth corner
            landmarks[54]     # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (assuming standard webcam)
        h, w = image_shape
        focal_length = w
        center = (w // 2, h // 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4, 1))
        
        try:
            # Solve PnP to get rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                pose_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
                
                return {
                    "yaw": pose_angles[1],      # Left-right rotation
                    "pitch": pose_angles[0],    # Up-down rotation
                    "roll": pose_angles[2],     # Tilt rotation
                    "confidence": 1.0,
                    "rotation_vector": rotation_vector,
                    "translation_vector": translation_vector
                }
            else:
                return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.0}
                
        except Exception as e:
            logger.warning(f"Pose estimation failed: {e}")
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.0}
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (in radians)."""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        return (x, y, z)


class FaceProfile:
    """Comprehensive face profile with quality metrics and pose information."""
    
    def __init__(self, face: Face, image: np.ndarray, landmarks: Optional[np.ndarray] = None):
        self.face = face
        self.image = image
        self.landmarks = landmarks
        self.quality_metrics = {}
        self.pose_info = {}
        self.enhancement_suggestions = []
        
        # Calculate all metrics
        self._calculate_quality_metrics()
        self._estimate_pose()
        self._generate_enhancement_suggestions()
    
    def _calculate_quality_metrics(self):
        """Calculate comprehensive quality metrics."""
        metrics = FaceQualityMetrics()
        
        self.quality_metrics = {
            "sharpness": metrics.calculate_sharpness(self.image),
            "lighting": metrics.calculate_lighting_quality(self.image),
            "resolution": metrics.calculate_resolution_quality(self.image),
            "pose": metrics.calculate_pose_quality(self.landmarks),
            "artifacts": metrics.detect_artifacts(self.image)
        }
        
        # Calculate overall quality score
        weights = {
            "sharpness": 0.25,
            "lighting": 0.20,
            "resolution": 0.20,
            "pose": 0.25,
            "artifacts": 0.10
        }
        
        self.quality_metrics["overall"] = sum(
            self.quality_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
    
    def _estimate_pose(self):
        """Estimate face pose."""
        if self.landmarks is not None:
            pose_estimator = PoseEstimator()
            self.pose_info = pose_estimator.estimate_pose(
                self.landmarks, self.image.shape[:2]
            )
        else:
            self.pose_info = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.0}
    
    def _generate_enhancement_suggestions(self):
        """Generate suggestions for improving face quality."""
        suggestions = []
        
        if self.quality_metrics["sharpness"] < 0.5:
            suggestions.append("Consider sharpening the image")
        
        if self.quality_metrics["lighting"] < 0.4:
            suggestions.append("Improve lighting conditions")
        
        if self.quality_metrics["resolution"] < 0.6:
            suggestions.append("Use higher resolution image")
        
        if self.quality_metrics["pose"] < 0.7:
            suggestions.append("Use more frontal face pose")
        
        if self.quality_metrics["artifacts"] < 0.7:
            suggestions.append("Reduce compression artifacts")
        
        self.enhancement_suggestions = suggestions
    
    def get_quality_score(self) -> float:
        """Get overall quality score (0-1)."""
        return self.quality_metrics["overall"]
    
    def is_suitable_for_swapping(self, min_quality: float = 0.6) -> bool:
        """Check if face is suitable for swapping based on quality."""
        return self.get_quality_score() >= min_quality
    
    def get_frontality_score(self) -> float:
        """Get frontality score based on pose."""
        yaw = abs(self.pose_info.get("yaw", 0))
        pitch = abs(self.pose_info.get("pitch", 0))
        
        # Convert radians to degrees and normalize
        yaw_deg = math.degrees(yaw)
        pitch_deg = math.degrees(pitch)
        
        # Good frontality is within 15 degrees
        max_angle = 15.0
        yaw_score = max(0, 1 - yaw_deg / max_angle)
        pitch_score = max(0, 1 - pitch_deg / max_angle)
        
        return (yaw_score + pitch_score) / 2.0


class EnhancedFaceProfiler:
    """Enhanced face profiler with advanced analysis capabilities."""
    
    def __init__(self):
        self.quality_threshold = 0.6
        self.pose_estimator = PoseEstimator()
        self.detected_faces = []
        self.face_profiles = []
    
    def analyze_faces_in_frame(self, frame: Frame, faces: List[Face]) -> List[FaceProfile]:
        """
        Analyze all faces in a frame and create comprehensive profiles.
        
        Args:
            frame: Input frame
            faces: List of detected faces
            
        Returns:
            List of face profiles with quality metrics
        """
        profiles = []
        
        for face in faces:
            try:
                # Extract face region
                face_image = self._extract_face_region(frame, face)
                
                # Get landmarks if available
                landmarks = getattr(face, 'landmark_2d_106', None)
                if landmarks is None:
                    landmarks = getattr(face, 'kps', None)
                
                # Create profile
                profile = FaceProfile(face, face_image, landmarks)
                profiles.append(profile)
                
            except Exception as e:
                logger.warning(f"Failed to analyze face: {e}")
                continue
        
        # Sort by quality score
        profiles.sort(key=lambda p: p.get_quality_score(), reverse=True)
        
        return profiles
    
    def _extract_face_region(self, frame: Frame, face: Face) -> np.ndarray:
        """Extract face region from frame."""
        if hasattr(face, 'bbox'):
            x1, y1, x2, y2 = map(int, face.bbox)
            
            # Add padding
            padding = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            return frame[y1:y2, x1:x2]
        else:
            # Fallback: return entire frame
            return frame
    
    def select_best_faces(self, profiles: List[FaceProfile], max_faces: int = 5) -> List[FaceProfile]:
        """
        Select the best faces based on quality metrics.
        
        Args:
            profiles: List of face profiles
            max_faces: Maximum number of faces to select
            
        Returns:
            List of best face profiles
        """
        # Filter by minimum quality
        suitable_faces = [p for p in profiles if p.is_suitable_for_swapping(self.quality_threshold)]
        
        # Sort by combined score (quality + frontality)
        suitable_faces.sort(
            key=lambda p: p.get_quality_score() * 0.7 + p.get_frontality_score() * 0.3,
            reverse=True
        )
        
        return suitable_faces[:max_faces]
    
    def generate_face_variations(self, profile: FaceProfile) -> List[np.ndarray]:
        """
        Generate face variations for better matching.
        
        Args:
            profile: Face profile to generate variations from
            
        Returns:
            List of face image variations
        """
        variations = [profile.image]
        
        try:
            # Brightness variations
            for factor in [0.8, 1.2]:
                bright_img = cv2.convertScaleAbs(profile.image, alpha=factor, beta=0)
                variations.append(bright_img)
            
            # Contrast variations
            for factor in [0.9, 1.1]:
                contrast_img = cv2.convertScaleAbs(profile.image, alpha=factor, beta=10)
                variations.append(contrast_img)
            
            # Slight rotations (for better pose matching)
            h, w = profile.image.shape[:2]
            center = (w // 2, h // 2)
            
            for angle in [-5, 5]:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(profile.image, M, (w, h))
                variations.append(rotated)
            
            # Histogram equalization for lighting normalization
            lab = cv2.cvtColor(profile.image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = cv2.equalizeHist(l)
            eq_img = cv2.merge([l_eq, a, b])
            eq_img = cv2.cvtColor(eq_img, cv2.COLOR_LAB2BGR)
            variations.append(eq_img)
            
        except Exception as e:
            logger.warning(f"Failed to generate face variations: {e}")
        
        return variations
    
    def create_multi_angle_profile(self, profiles: List[FaceProfile]) -> Dict[str, Any]:
        """
        Create a multi-angle face profile from multiple face detections.
        
        Args:
            profiles: List of face profiles from different angles
            
        Returns:
            Dictionary containing multi-angle profile data
        """
        if not profiles:
            return {}
        
        # Group by pose angles
        frontal_faces = []
        left_profile = []
        right_profile = []
        
        for profile in profiles:
            yaw = profile.pose_info.get("yaw", 0)
            yaw_deg = math.degrees(yaw)
            
            if abs(yaw_deg) < 15:
                frontal_faces.append(profile)
            elif yaw_deg > 15:
                left_profile.append(profile)
            elif yaw_deg < -15:
                right_profile.append(profile)
        
        # Select best from each category
        multi_angle_profile = {
            "frontal": frontal_faces[0] if frontal_faces else None,
            "left_profile": left_profile[0] if left_profile else None,
            "right_profile": right_profile[0] if right_profile else None,
            "all_profiles": profiles,
            "quality_stats": {
                "average_quality": np.mean([p.get_quality_score() for p in profiles]),
                "best_quality": max([p.get_quality_score() for p in profiles]),
                "pose_coverage": len([cat for cat in [frontal_faces, left_profile, right_profile] if cat])
            }
        }
        
        return multi_angle_profile


# Global enhanced face profiler instance
_enhanced_profiler = None


def get_enhanced_face_profiler() -> EnhancedFaceProfiler:
    """Get the global enhanced face profiler instance."""
    global _enhanced_profiler
    if _enhanced_profiler is None:
        _enhanced_profiler = EnhancedFaceProfiler()
    return _enhanced_profiler


def analyze_face_quality(face_image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Convenience function to analyze face quality.
    
    Args:
        face_image: Face image to analyze
        landmarks: Optional facial landmarks
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = FaceQualityMetrics()
    return {
        "sharpness": metrics.calculate_sharpness(face_image),
        "lighting": metrics.calculate_lighting_quality(face_image),
        "resolution": metrics.calculate_resolution_quality(face_image),
        "pose": metrics.calculate_pose_quality(landmarks),
        "artifacts": metrics.detect_artifacts(face_image)
    }