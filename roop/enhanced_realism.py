#!/usr/bin/env python3
"""
Enhanced face processing with hair integration and improved realism.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class HairRegionDetector:
    """Detect and segment hair regions for enhanced face processing."""
    
    def __init__(self):
        self.hair_cascade = None
        self.segmentation_model = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize hair detection models."""
        try:
            # In practice, load actual hair segmentation models
            # For now, we'll use mock initialization
            logger.info("Hair region detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize hair models: {e}")
            
    def detect_hair_region(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Detect hair region around face.
        
        Args:
            image: Input image as numpy array
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Hair mask as binary image, or None if detection fails
        """
        try:
            x, y, w, h = face_bbox
            
            # Expand region to include potential hair area
            hair_expansion_factor = 1.8
            expanded_x = max(0, int(x - w * (hair_expansion_factor - 1) / 2))
            expanded_y = max(0, int(y - h * (hair_expansion_factor - 1) / 2))
            expanded_w = min(image.shape[1] - expanded_x, int(w * hair_expansion_factor))
            expanded_h = min(image.shape[0] - expanded_y, int(h * hair_expansion_factor))
            
            # Extract expanded region
            expanded_region = image[expanded_y:expanded_y + expanded_h, 
                                 expanded_x:expanded_x + expanded_w]
            
            # Mock hair detection using color and texture analysis
            hair_mask = self._segment_hair_region(expanded_region)
            
            # Create full-size mask
            full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            full_mask[expanded_y:expanded_y + expanded_h, 
                     expanded_x:expanded_x + expanded_w] = hair_mask
            
            return full_mask
            
        except Exception as e:
            logger.error(f"Hair detection failed: {e}")
            return None
            
    def _segment_hair_region(self, region: np.ndarray) -> np.ndarray:
        """
        Segment hair region using color and texture analysis.
        
        Args:
            region: Image region to analyze
            
        Returns:
            Binary mask of hair region
        """
        # Convert to different color spaces for better hair detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        
        # Hair typically has lower saturation and specific hue ranges
        # Create initial mask based on color characteristics
        hair_mask = np.zeros(region.shape[:2], dtype=np.uint8)
        
        # Dark hair detection (low value, low saturation)
        dark_hair = cv2.inRange(hsv, (0, 0, 0), (180, 100, 100))
        
        # Blonde/light hair detection (higher saturation in yellow range)
        light_hair = cv2.inRange(hsv, (10, 30, 100), (30, 255, 255))
        
        # Combine masks
        hair_mask = cv2.bitwise_or(dark_hair, light_hair)
        
        # Refine using texture analysis
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Hair typically has high texture variation
        texture_features = self._analyze_texture(gray)
        texture_mask = (texture_features > np.percentile(texture_features, 70)).astype(np.uint8) * 255
        
        # Combine color and texture information
        combined_mask = cv2.bitwise_and(hair_mask, texture_mask)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
        
    def _analyze_texture(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Analyze texture characteristics of the image.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Texture feature map
        """
        # Use Gabor filters for texture analysis
        gabor_responses = []
        
        for theta in [0, 45, 90, 135]:  # Different orientations
            for frequency in [0.1, 0.3, 0.5]:  # Different frequencies
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 
                                          2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                response = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
                gabor_responses.append(response)
        
        # Combine responses
        texture_strength = np.mean(gabor_responses, axis=0)
        
        return texture_strength
        
    def refine_hair_mask(self, hair_mask: np.ndarray, face_landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """
        Refine hair mask using facial landmarks.
        
        Args:
            hair_mask: Initial hair mask
            face_landmarks: Facial landmark points
            
        Returns:
            Refined hair mask
        """
        refined_mask = hair_mask.copy()
        
        if face_landmarks:
            # Create face contour to exclude face region from hair
            face_contour = np.array(face_landmarks[:17], dtype=np.int32)  # Jawline
            face_mask = np.zeros_like(hair_mask)
            cv2.fillPoly(face_mask, [face_contour], 255)
            
            # Subtract face region from hair mask
            refined_mask = cv2.bitwise_and(refined_mask, cv2.bitwise_not(face_mask))
        
        return refined_mask


class EnhancedFaceMapper:
    """Enhanced face mapping with angle adaptation and obstruction handling."""
    
    def __init__(self):
        self.landmark_detector = None
        self.pose_estimator = None
        self.obstruction_detector = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize face mapping models."""
        try:
            # Mock initialization - in practice, load actual models
            logger.info("Enhanced face mapper initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize face mapping models: {e}")
            
    def map_face_with_pose_adaptation(self, source_face: np.ndarray, target_face: np.ndarray,
                                    source_landmarks: List[Tuple[int, int]],
                                    target_landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """
        Map source face to target with pose adaptation.
        
        Args:
            source_face: Source face image
            target_face: Target face image
            source_landmarks: Source face landmarks
            target_landmarks: Target face landmarks
            
        Returns:
            Mapped face image
        """
        try:
            # Estimate pose angles
            source_pose = self._estimate_pose(source_landmarks)
            target_pose = self._estimate_pose(target_landmarks)
            
            # Calculate pose difference
            pose_diff = {
                'yaw': target_pose['yaw'] - source_pose['yaw'],
                'pitch': target_pose['pitch'] - source_pose['pitch'],
                'roll': target_pose['roll'] - source_pose['roll']
            }
            
            # Apply pose compensation
            compensated_source = self._apply_pose_compensation(source_face, source_landmarks, pose_diff)
            
            # Perform enhanced mapping
            mapped_face = self._perform_enhanced_mapping(compensated_source, target_face,
                                                       source_landmarks, target_landmarks)
            
            return mapped_face
            
        except Exception as e:
            logger.error(f"Face mapping with pose adaptation failed: {e}")
            return target_face
            
    def _estimate_pose(self, landmarks: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        Estimate face pose from landmarks.
        
        Args:
            landmarks: Facial landmark points
            
        Returns:
            Pose angles (yaw, pitch, roll) in degrees
        """
        if len(landmarks) < 68:
            return {'yaw': 0, 'pitch': 0, 'roll': 0}
            
        # Use key landmarks for pose estimation
        nose_tip = landmarks[30]
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        left_mouth_corner = landmarks[48]
        right_mouth_corner = landmarks[54]
        
        # Calculate yaw (left-right rotation)
        eye_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
        nose_x = nose_tip[0]
        face_center_x = (left_mouth_corner[0] + right_mouth_corner[0]) / 2
        
        # Simplified yaw calculation
        yaw = np.arctan2(nose_x - face_center_x, abs(left_eye_corner[0] - right_eye_corner[0])) * 180 / np.pi
        
        # Calculate pitch (up-down rotation)
        eye_center_y = (left_eye_corner[1] + right_eye_corner[1]) / 2
        nose_y = nose_tip[1]
        mouth_center_y = (left_mouth_corner[1] + right_mouth_corner[1]) / 2
        
        pitch = np.arctan2(nose_y - eye_center_y, mouth_center_y - eye_center_y) * 180 / np.pi
        
        # Calculate roll (tilt rotation)
        roll = np.arctan2(right_eye_corner[1] - left_eye_corner[1], 
                         right_eye_corner[0] - left_eye_corner[0]) * 180 / np.pi
        
        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}
        
    def _apply_pose_compensation(self, face_image: np.ndarray, landmarks: List[Tuple[int, int]],
                               pose_diff: Dict[str, float]) -> np.ndarray:
        """
        Apply pose compensation to face image.
        
        Args:
            face_image: Face image to compensate
            landmarks: Facial landmarks
            pose_diff: Pose difference to compensate
            
        Returns:
            Pose-compensated face image
        """
        compensated = face_image.copy()
        
        # Apply rotation compensation
        if abs(pose_diff['roll']) > 2:  # Only compensate significant roll
            center = (face_image.shape[1] // 2, face_image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -pose_diff['roll'], 1.0)
            compensated = cv2.warpAffine(compensated, rotation_matrix, 
                                       (face_image.shape[1], face_image.shape[0]))
        
        # Apply perspective transformation for yaw/pitch compensation
        if abs(pose_diff['yaw']) > 5 or abs(pose_diff['pitch']) > 5:
            compensated = self._apply_perspective_correction(compensated, pose_diff)
        
        return compensated
        
    def _apply_perspective_correction(self, face_image: np.ndarray, 
                                    pose_diff: Dict[str, float]) -> np.ndarray:
        """
        Apply perspective correction for pose compensation.
        
        Args:
            face_image: Face image
            pose_diff: Pose differences
            
        Returns:
            Perspective-corrected image
        """
        h, w = face_image.shape[:2]
        
        # Define source points (corners of the face)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Calculate destination points based on pose differences
        yaw_factor = pose_diff['yaw'] / 90.0  # Normalize to [-1, 1]
        pitch_factor = pose_diff['pitch'] / 90.0
        
        # Adjust destination points for perspective transformation
        offset_x = int(w * 0.1 * yaw_factor)
        offset_y = int(h * 0.1 * pitch_factor)
        
        dst_points = np.float32([
            [offset_x, offset_y],
            [w - offset_x, offset_y],
            [w + offset_x, h - offset_y],
            [-offset_x, h - offset_y]
        ])
        
        # Apply perspective transformation
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        corrected = cv2.warpPerspective(face_image, perspective_matrix, (w, h))
        
        return corrected
        
    def _perform_enhanced_mapping(self, source_face: np.ndarray, target_face: np.ndarray,
                                source_landmarks: List[Tuple[int, int]],
                                target_landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """
        Perform enhanced face mapping with improved blending.
        
        Args:
            source_face: Source face image
            target_face: Target face image  
            source_landmarks: Source landmarks
            target_landmarks: Target landmarks
            
        Returns:
            Enhanced mapped face
        """
        # Calculate transformation matrix
        transform_matrix = self._calculate_transform_matrix(source_landmarks, target_landmarks)
        
        # Warp source face to target pose
        warped_source = cv2.warpAffine(source_face, transform_matrix, 
                                     (target_face.shape[1], target_face.shape[0]))
        
        # Create blending mask
        blending_mask = self._create_enhanced_blending_mask(target_landmarks, target_face.shape)
        
        # Apply multi-scale blending
        blended_face = self._multi_scale_blending(warped_source, target_face, blending_mask)
        
        return blended_face
        
    def _calculate_transform_matrix(self, source_landmarks: List[Tuple[int, int]],
                                  target_landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Calculate transformation matrix between landmark sets."""
        if len(source_landmarks) < 3 or len(target_landmarks) < 3:
            return np.eye(2, 3, dtype=np.float32)
            
        # Use key landmarks for transformation
        key_indices = [30, 36, 45]  # Nose tip, left eye corner, right eye corner
        
        src_points = np.array([source_landmarks[i] for i in key_indices], dtype=np.float32)
        dst_points = np.array([target_landmarks[i] for i in key_indices], dtype=np.float32)
        
        # Calculate affine transformation
        transform_matrix = cv2.getAffineTransform(src_points, dst_points)
        
        return transform_matrix
        
    def _create_enhanced_blending_mask(self, landmarks: List[Tuple[int, int]], 
                                     image_shape: Tuple[int, int]) -> np.ndarray:
        """Create enhanced blending mask with smooth transitions."""
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        
        if len(landmarks) >= 68:
            # Create face contour
            face_contour = np.array(landmarks[:17] + landmarks[26:16:-1], dtype=np.int32)
            cv2.fillPoly(mask, [face_contour], 1.0)
            
            # Apply Gaussian blur for smooth transitions
            mask = cv2.GaussianBlur(mask, (21, 21), 10)
            
            # Create feathered edges
            mask = np.clip(mask, 0, 1)
        
        return mask
        
    def _multi_scale_blending(self, source: np.ndarray, target: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
        """Apply multi-scale blending for natural results."""
        # Convert to float for blending
        source_f = source.astype(np.float32) / 255.0
        target_f = target.astype(np.float32) / 255.0
        
        # Create Gaussian pyramids
        source_pyramid = self._create_gaussian_pyramid(source_f, 5)
        target_pyramid = self._create_gaussian_pyramid(target_f, 5)
        mask_pyramid = self._create_gaussian_pyramid(mask, 5)
        
        # Blend each level
        blended_pyramid = []
        for i in range(len(source_pyramid)):
            level_mask = mask_pyramid[i]
            if len(level_mask.shape) == 2:
                level_mask = level_mask[:, :, np.newaxis]
                
            blended_level = source_pyramid[i] * level_mask + target_pyramid[i] * (1 - level_mask)
            blended_pyramid.append(blended_level)
        
        # Reconstruct from pyramid
        blended = self._reconstruct_from_pyramid(blended_pyramid)
        
        # Convert back to uint8
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        
        return blended
        
    def _create_gaussian_pyramid(self, image: np.ndarray, levels: int) -> List[np.ndarray]:
        """Create Gaussian pyramid."""
        pyramid = [image]
        
        for i in range(levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
            
        return pyramid
        
    def _reconstruct_from_pyramid(self, pyramid: List[np.ndarray]) -> np.ndarray:
        """Reconstruct image from Gaussian pyramid."""
        image = pyramid[-1]
        
        for i in range(len(pyramid) - 2, -1, -1):
            image = cv2.pyrUp(image)
            
            # Ensure same size
            target_shape = pyramid[i].shape[:2]
            if image.shape[:2] != target_shape:
                image = cv2.resize(image, (target_shape[1], target_shape[0]))
                
            image = cv2.add(image, pyramid[i])
            
        return image


class ObstructionHandler:
    """Handle facial obstructions like glasses, masks, etc."""
    
    def __init__(self):
        self.obstruction_detectors = {}
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize obstruction detection models."""
        try:
            # Mock initialization - in practice, load actual models
            self.obstruction_detectors = {
                'glasses': None,
                'sunglasses': None,
                'mask': None,
                'scarf': None,
                'hat': None
            }
            logger.info("Obstruction detectors initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize obstruction detectors: {e}")
            
    def detect_obstructions(self, face_image: np.ndarray, 
                          landmarks: List[Tuple[int, int]]) -> Dict[str, any]:
        """
        Detect various types of facial obstructions.
        
        Args:
            face_image: Face image
            landmarks: Facial landmarks
            
        Returns:
            Dictionary of detected obstructions and their properties
        """
        obstructions = {}
        
        # Detect glasses
        glasses_info = self._detect_glasses(face_image, landmarks)
        if glasses_info:
            obstructions['glasses'] = glasses_info
            
        # Detect masks
        mask_info = self._detect_mask(face_image, landmarks)
        if mask_info:
            obstructions['mask'] = mask_info
            
        # Detect other accessories
        hat_info = self._detect_hat(face_image, landmarks)
        if hat_info:
            obstructions['hat'] = hat_info
            
        return obstructions
        
    def _detect_glasses(self, face_image: np.ndarray, 
                       landmarks: List[Tuple[int, int]]) -> Optional[Dict]:
        """Detect eyeglasses or sunglasses."""
        if len(landmarks) < 68:
            return None
            
        # Eye regions
        left_eye_region = landmarks[36:42]
        right_eye_region = landmarks[42:48]
        
        # Extract eye regions with padding
        eye_regions = self._extract_eye_regions(face_image, left_eye_region, right_eye_region)
        
        # Analyze for glass-like features (edges, reflections)
        glasses_confidence = 0.0
        
        for eye_region in eye_regions:
            # Edge detection
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_eye, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Check for rectangular structures (glasses frames)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Check if contour resembles glasses frame
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) >= 4:  # Rectangular-ish shape
                    glasses_confidence += 0.3
                    
        # Check for reflections (characteristic of glasses)
        reflection_score = self._detect_reflections(face_image, left_eye_region + right_eye_region)
        glasses_confidence += reflection_score * 0.4
        
        if glasses_confidence > 0.5:
            return {
                'type': 'sunglasses' if reflection_score > 0.7 else 'glasses',
                'confidence': glasses_confidence,
                'regions': [left_eye_region, right_eye_region]
            }
            
        return None
        
    def _detect_mask(self, face_image: np.ndarray, 
                    landmarks: List[Tuple[int, int]]) -> Optional[Dict]:
        """Detect face masks."""
        if len(landmarks) < 68:
            return None
            
        # Nose and mouth region
        nose_landmarks = landmarks[27:36]
        mouth_landmarks = landmarks[48:68]
        
        # Check for coverage in nose/mouth area
        mask_region = self._extract_mask_region(face_image, nose_landmarks, mouth_landmarks)
        
        # Analyze color uniformity (masks often have uniform color)
        color_uniformity = self._analyze_color_uniformity(mask_region)
        
        # Check for fabric texture
        texture_score = self._analyze_fabric_texture(mask_region)
        
        mask_confidence = (color_uniformity * 0.4 + texture_score * 0.6)
        
        if mask_confidence > 0.6:
            return {
                'type': 'mask',
                'confidence': mask_confidence,
                'region': nose_landmarks + mouth_landmarks
            }
            
        return None
        
    def _detect_hat(self, face_image: np.ndarray, 
                   landmarks: List[Tuple[int, int]]) -> Optional[Dict]:
        """Detect hats or head coverings."""
        if len(landmarks) < 68:
            return None
            
        # Check area above forehead
        forehead_landmarks = landmarks[17:27]
        
        # Expand upward to check for hat
        hat_region = self._extract_hat_region(face_image, forehead_landmarks)
        
        if hat_region is None:
            return None
            
        # Analyze for hat-like features
        hat_confidence = self._analyze_hat_features(hat_region)
        
        if hat_confidence > 0.5:
            return {
                'type': 'hat',
                'confidence': hat_confidence,
                'region': hat_region
            }
            
        return None
        
    def _extract_eye_regions(self, face_image: np.ndarray, 
                           left_eye: List[Tuple[int, int]], 
                           right_eye: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Extract eye regions with padding."""
        regions = []
        
        for eye_landmarks in [left_eye, right_eye]:
            if not eye_landmarks:
                continue
                
            # Calculate bounding box
            x_coords = [p[0] for p in eye_landmarks]
            y_coords = [p[1] for p in eye_landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(face_image.shape[1], x_max + padding)
            y_max = min(face_image.shape[0], y_max + padding)
            
            eye_region = face_image[y_min:y_max, x_min:x_max]
            regions.append(eye_region)
            
        return regions
        
    def _extract_mask_region(self, face_image: np.ndarray,
                           nose_landmarks: List[Tuple[int, int]],
                           mouth_landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Extract potential mask region."""
        all_landmarks = nose_landmarks + mouth_landmarks
        
        if not all_landmarks:
            return np.array([])
            
        x_coords = [p[0] for p in all_landmarks]
        y_coords = [p[1] for p in all_landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Expand region
        expansion = 20
        x_min = max(0, x_min - expansion)
        y_min = max(0, y_min - expansion)
        x_max = min(face_image.shape[1], x_max + expansion)
        y_max = min(face_image.shape[0], y_max + expansion)
        
        return face_image[y_min:y_max, x_min:x_max]
        
    def _extract_hat_region(self, face_image: np.ndarray,
                          forehead_landmarks: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """Extract potential hat region."""
        if not forehead_landmarks:
            return None
            
        # Find top of forehead
        y_coords = [p[1] for p in forehead_landmarks]
        x_coords = [p[0] for p in forehead_landmarks]
        
        forehead_top = min(y_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        
        # Check region above forehead
        hat_height = int((forehead_top) * 0.5)  # Check upper half above forehead
        
        if hat_height < 20:  # Too small to be meaningful
            return None
            
        hat_region = face_image[0:forehead_top, x_min:x_max]
        
        return hat_region
        
    def _detect_reflections(self, image: np.ndarray, eye_landmarks: List[Tuple[int, int]]) -> float:
        """Detect reflections characteristic of glasses."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find bright spots (potential reflections)
        _, bright_areas = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Count bright pixels in eye region
        eye_mask = np.zeros_like(gray)
        if eye_landmarks:
            eye_contour = np.array(eye_landmarks, dtype=np.int32)
            cv2.fillPoly(eye_mask, [eye_contour], 255)
            
            bright_in_eyes = cv2.bitwise_and(bright_areas, eye_mask)
            reflection_ratio = np.sum(bright_in_eyes > 0) / np.sum(eye_mask > 0)
            
            return min(reflection_ratio * 5, 1.0)  # Scale and cap at 1.0
            
        return 0.0
        
    def _analyze_color_uniformity(self, region: np.ndarray) -> float:
        """Analyze color uniformity in a region."""
        if region.size == 0:
            return 0.0
            
        # Calculate color variance
        region_float = region.astype(np.float32)
        color_variance = np.var(region_float, axis=(0, 1))
        
        # Lower variance indicates more uniform color
        uniformity = 1.0 / (1.0 + np.mean(color_variance) / 100)
        
        return uniformity
        
    def _analyze_fabric_texture(self, region: np.ndarray) -> float:
        """Analyze fabric-like texture."""
        if region.size == 0:
            return 0.0
            
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Use LBP (Local Binary Pattern) for texture analysis
        texture_features = self._calculate_lbp(gray)
        
        # Fabric typically has specific texture patterns
        # This is a simplified analysis
        texture_variance = np.var(texture_features)
        fabric_score = min(texture_variance / 1000, 1.0)
        
        return fabric_score
        
    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern features."""
        # Simplified LBP implementation
        rows, cols = gray_image.shape
        lbp = np.zeros_like(gray_image)
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = gray_image[i, j]
                
                # Check 8 neighbors
                pattern = 0
                pattern |= (gray_image[i-1, j-1] >= center) << 7
                pattern |= (gray_image[i-1, j] >= center) << 6
                pattern |= (gray_image[i-1, j+1] >= center) << 5
                pattern |= (gray_image[i, j+1] >= center) << 4
                pattern |= (gray_image[i+1, j+1] >= center) << 3
                pattern |= (gray_image[i+1, j] >= center) << 2
                pattern |= (gray_image[i+1, j-1] >= center) << 1
                pattern |= (gray_image[i, j-1] >= center) << 0
                
                lbp[i, j] = pattern
                
        return lbp
        
    def _analyze_hat_features(self, hat_region: np.ndarray) -> float:
        """Analyze features indicative of hats."""
        if hat_region.size == 0:
            return 0.0
            
        # Check for horizontal edges (hat brims)
        gray = cv2.cvtColor(hat_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=20, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 15 or angle > 165:  # Nearly horizontal
                    horizontal_lines += 1
                    
        # Color uniformity (hats often have uniform color)
        color_uniformity = self._analyze_color_uniformity(hat_region)
        
        # Combine features
        hat_score = (horizontal_lines / 10) * 0.6 + color_uniformity * 0.4
        
        return min(hat_score, 1.0)