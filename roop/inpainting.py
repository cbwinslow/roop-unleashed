"""
Inpainting module for face correction and enhancement.
Integrates stable diffusion and traditional inpainting techniques.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from abc import ABC, abstractmethod

from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path

logger = logging.getLogger(__name__)


class InpaintingMaskGenerator:
    """Generates masks for inpainting operations."""
    
    @staticmethod
    def create_face_boundary_mask(face: Face, frame_shape: Tuple[int, int, int], 
                                 expand_ratio: float = 0.1) -> np.ndarray:
        """
        Create mask around face boundaries for seamless blending.
        
        Args:
            face: Face object with landmarks
            frame_shape: Shape of the target frame (H, W, C)
            expand_ratio: How much to expand the face region
            
        Returns:
            Binary mask for inpainting
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if hasattr(face, 'bbox'):
            x1, y1, x2, y2 = map(int, face.bbox)
            
            # Expand the bounding box
            expand_x = int((x2 - x1) * expand_ratio)
            expand_y = int((y2 - y1) * expand_ratio)
            
            x1 = max(0, x1 - expand_x)
            y1 = max(0, y1 - expand_y)
            x2 = min(w, x2 + expand_x)
            y2 = min(h, y2 + expand_y)
            
            # Create elliptical mask for natural blending
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
    
    @staticmethod
    def create_occlusion_mask(frame: Frame, text_prompt: str = None) -> np.ndarray:
        """
        Create mask for occluded regions (glasses, masks, hair, etc.).
        
        Args:
            frame: Input frame
            text_prompt: Text description of what to mask (optional)
            
        Returns:
            Binary mask for occluded regions
        """
        # Basic implementation using traditional CV methods
        # In a full implementation, this would use CLIP or similar for text-based masking
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect dark regions that might be glasses/masks
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Detect edge regions that might be hair/accessories
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(dark_mask, edges_dilated)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask


class BaseInpainter(ABC):
    """Abstract base class for inpainting methods."""
    
    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        """Perform inpainting on the given image and mask."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the inpainting method is available."""
        pass


class TraditionalInpainter(BaseInpainter):
    """Traditional CV-based inpainting using OpenCV methods."""
    
    def __init__(self, method: str = "telea"):
        """
        Initialize traditional inpainter.
        
        Args:
            method: 'telea' or 'ns' (Navier-Stokes)
        """
        self.method = method
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform traditional inpainting.
        
        Args:
            image: Input image
            mask: Binary mask (255 for regions to inpaint)
            **kwargs: Additional parameters (inpaint_radius, etc.)
            
        Returns:
            Inpainted image
        """
        inpaint_radius = kwargs.get('inpaint_radius', 3)
        
        if self.method.lower() == 'telea':
            method_flag = cv2.INPAINT_TELEA
        else:
            method_flag = cv2.INPAINT_NS
        
        return cv2.inpaint(image, mask, inpaint_radius, method_flag)
    
    def is_available(self) -> bool:
        """Traditional inpainting is always available with OpenCV."""
        return True


class AdvancedObstructionDetector:
    """Advanced obstruction detection using multiple techniques."""
    
    def __init__(self):
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
    
    def detect_glasses(self, frame: Frame, face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Detect glasses in the face region."""
        x1, y1, x2, y2 = face_bbox
        face_region = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Detect circular/elliptical shapes (lens frames)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=50
        )
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Create mask for detected circles
                cv2.circle(mask, (x, y), r + 5, 255, -1)
        
        # Also detect rectangular frames
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Check if contour could be glasses frame
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Reasonable size for glasses
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Resize mask to original frame size
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask
        
        return full_mask
    
    def detect_hair_occlusion(self, frame: Frame, face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Detect hair covering parts of the face."""
        x1, y1, x2, y2 = face_bbox
        
        # Expand region to include hair
        h, w = frame.shape[:2]
        hair_y1 = max(0, y1 - (y2 - y1) // 2)  # Extend upward
        hair_x1 = max(0, x1 - (x2 - x1) // 4)  # Extend sides
        hair_x2 = min(w, x2 + (x2 - x1) // 4)
        
        hair_region = frame[hair_y1:y2, hair_x1:hair_x2]
        
        # Convert to HSV for better hair detection
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-skin colors (likely hair)
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        hair_mask = cv2.bitwise_not(skin_mask)
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
        
        # Focus on face region only
        face_hair_mask = hair_mask[y1-hair_y1:y2-hair_y1, x1-hair_x1:x2-hair_x1]
        
        # Resize to full frame
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = face_hair_mask
        
        return full_mask
    
    def detect_face_mask(self, frame: Frame, face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Detect face masks or coverings."""
        x1, y1, x2, y2 = face_bbox
        face_region = frame[y1:y2, x1:x2]
        
        # Focus on lower half of face where masks typically are
        lower_half_y = (y2 - y1) // 2
        lower_face = face_region[lower_half_y:, :]
        
        # Convert to HSV
        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        
        # Detect non-skin colors in lower face
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        mask_mask = cv2.bitwise_not(skin_mask)
        
        # Look for edges that might indicate mask boundaries
        gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine color and edge information
        combined = cv2.bitwise_or(mask_mask, edges)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Create full face mask
        face_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        face_mask[lower_half_y:, :] = combined
        
        # Resize to full frame
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = face_mask
        
        return full_mask
    
    def detect_all_obstructions(self, frame: Frame, face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Detect all types of obstructions."""
        glasses_mask = self.detect_glasses(frame, face_bbox)
        hair_mask = self.detect_hair_occlusion(frame, face_bbox)
        mask_mask = self.detect_face_mask(frame, face_bbox)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(glasses_mask, hair_mask)
        combined_mask = cv2.bitwise_or(combined_mask, mask_mask)
        
        return combined_mask


class EdgeAwareInpainter(BaseInpainter):
    """Edge-aware inpainting that preserves facial structure."""
    
    def __init__(self):
        self.patch_size = 9
        
    def inpaint(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform edge-aware inpainting.
        
        Args:
            image: Input image
            mask: Binary mask (255 for regions to inpaint)
            
        Returns:
            Inpainted image
        """
        # Start with traditional inpainting
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # Enhance with edge-aware techniques
        result = self._edge_aware_enhancement(image, result, mask)
        
        return result
    
    def _edge_aware_enhancement(self, original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply edge-aware enhancement to inpainted regions."""
        # Detect edges in the original image
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_orig, 50, 150)
        
        # Dilate edges to create influence regions
        edge_influence = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        
        # Apply guided filtering near edges
        result = inpainted.copy()
        
        # Use bilateral filtering in inpainted regions near edges
        bilateral_filtered = cv2.bilateralFilter(inpainted, 9, 75, 75)
        
        # Blend based on edge proximity
        edge_regions = cv2.bitwise_and(mask, edge_influence)
        
        # Apply stronger filtering near edges
        result = np.where(edge_regions[..., np.newaxis] > 0, bilateral_filtered, result)
        
        return result
    
    def is_available(self) -> bool:
        """Edge-aware inpainting is always available."""
        return True


class ContextAwareInpainter(BaseInpainter):
    """Context-aware inpainting using surrounding facial features."""
    
    def __init__(self):
        self.search_radius = 20
        
    def inpaint(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform context-aware inpainting.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Inpainted image
        """
        result = image.copy()
        
        # Find contours of masked regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding box of masked region
            x, y, w, h = cv2.boundingRect(contour)
            
            # Inpaint this region using context
            region_result = self._inpaint_region(image, mask, (x, y, w, h))
            
            # Update result
            region_mask = mask[y:y+h, x:x+w]
            result[y:y+h, x:x+w] = np.where(
                region_mask[..., np.newaxis] > 0,
                region_result,
                result[y:y+h, x:x+w]
            )
        
        return result
    
    def _inpaint_region(self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Inpaint a specific region using contextual information."""
        x, y, w, h = bbox
        
        # Expand search area
        search_x1 = max(0, x - self.search_radius)
        search_y1 = max(0, y - self.search_radius)
        search_x2 = min(image.shape[1], x + w + self.search_radius)
        search_y2 = min(image.shape[0], y + h + self.search_radius)
        
        # Extract search region
        search_region = image[search_y1:search_y2, search_x1:search_x2]
        search_mask = mask[search_y1:search_y2, search_x1:search_x2]
        
        # Use PatchMatch-style inpainting
        result_region = cv2.inpaint(search_region, search_mask, 5, cv2.INPAINT_TELEA)
        
        # Extract the target region
        target_y1 = y - search_y1
        target_x1 = x - search_x1
        target_result = result_region[target_y1:target_y1+h, target_x1:target_x1+w]
        
        return target_result
    
    def is_available(self) -> bool:
        """Context-aware inpainting is always available."""
        return True


class StableDiffusionInpainter(BaseInpainter):
    """Stable Diffusion-based inpainting (placeholder for future implementation)."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize SD inpainter.
        
        Args:
            model_path: Path to stable diffusion model
        """
        self.model_path = model_path
        self.model = None
        self.pipeline = None
    
    def _load_model(self):
        """Load the stable diffusion inpainting model."""
        try:
            # Placeholder for actual SD model loading
            # In full implementation would use diffusers library:
            # from diffusers import StableDiffusionInpaintPipeline
            # self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(...)
            logger.info("Stable Diffusion inpainting model loading not yet implemented")
            return False
        except Exception as e:
            logger.warning(f"Failed to load SD inpainting model: {e}")
            return False
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform SD-based inpainting.
        
        Args:
            image: Input image
            mask: Binary mask
            **kwargs: Additional parameters (prompt, guidance_scale, etc.)
            
        Returns:
            Inpainted image
        """
        if not self.is_available():
            # Fallback to traditional inpainting
            fallback = TraditionalInpainter()
            return fallback.inpaint(image, mask, **kwargs)
        
        # Placeholder for actual SD inpainting
        prompt = kwargs.get('prompt', "natural face, high quality")
        guidance_scale = kwargs.get('guidance_scale', 7.5)
        
        # Convert to PIL format for SD pipeline
        # result = self.pipeline(image=image, mask_image=mask, prompt=prompt, ...)
        
        # For now, return traditional inpainting as fallback
        fallback = TraditionalInpainter()
        return fallback.inpaint(image, mask, **kwargs)
    
    def is_available(self) -> bool:
        """Check if SD inpainting is available."""
        return False  # Will be True when fully implemented


class InpaintingManager:
    """Manages different inpainting methods and provides unified interface."""
    
    def __init__(self):
        self.inpainters = {
            'traditional_telea': TraditionalInpainter('telea'),
            'traditional_ns': TraditionalInpainter('ns'),
            'edge_aware': EdgeAwareInpainter(),
            'context_aware': ContextAwareInpainter(),
            'stable_diffusion': StableDiffusionInpainter(),
        }
        self.default_method = 'edge_aware'  # Use enhanced method as default
        self.obstruction_detector = AdvancedObstructionDetector()
    
    def get_available_methods(self) -> List[str]:
        """Get list of available inpainting methods."""
        return [name for name, inpainter in self.inpainters.items() 
                if inpainter.is_available()]
    
    def detect_and_correct_obstructions(self, 
                                      frame: Frame, 
                                      face: Face,
                                      method: str = None) -> Tuple[Frame, np.ndarray]:
        """
        Automatically detect and correct face obstructions.
        
        Args:
            frame: Input frame
            face: Face object with detection info
            method: Inpainting method to use
            
        Returns:
            Tuple of (corrected_frame, obstruction_mask)
        """
        # Get face bounding box
        if hasattr(face, 'bbox'):
            face_bbox = tuple(map(int, face.bbox))
        else:
            # Fallback: use entire frame
            h, w = frame.shape[:2]
            face_bbox = (0, 0, w, h)
        
        # Detect all obstructions
        obstruction_mask = self.obstruction_detector.detect_all_obstructions(frame, face_bbox)
        
        if np.sum(obstruction_mask) == 0:
            # No obstructions detected
            return frame, obstruction_mask
        
        # Use intelligent method selection based on obstruction type
        if method is None:
            method = self._select_best_method(obstruction_mask)
        
        # Perform inpainting
        corrected_frame = self.inpaint_with_mask(frame, obstruction_mask, method)
        
        return corrected_frame, obstruction_mask
    
    def _select_best_method(self, mask: np.ndarray) -> str:
        """Select the best inpainting method based on mask characteristics."""
        # Analyze mask properties
        mask_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        coverage_ratio = mask_area / total_area
        
        # Count number of separate regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_regions = len(contours)
        
        # Select method based on characteristics
        if coverage_ratio > 0.3:  # Large area coverage
            return 'context_aware'
        elif num_regions > 5:  # Many small regions
            return 'edge_aware'
        else:  # Medium complexity
            return 'edge_aware'
    
    def inpaint_with_mask(self, frame: Frame, mask: np.ndarray, method: str = None) -> Frame:
        """
        Inpaint frame using provided mask.
        
        Args:
            frame: Input frame
            mask: Binary mask (255 for regions to inpaint)
            method: Inpainting method to use
            
        Returns:
            Inpainted frame
        """
        if method is None:
            method = self.default_method
        
        if method not in self.inpainters:
            logger.warning(f"Unknown inpainting method: {method}, using default")
            method = self.default_method
        
        inpainter = self.inpainters[method]
        if not inpainter.is_available():
            logger.warning(f"Inpainting method {method} not available, using fallback")
            inpainter = self.inpainters['traditional_telea']
        
        try:
            inpainted_frame = inpainter.inpaint(frame, mask)
            logger.debug(f"Successfully inpainted using {method}")
            return inpainted_frame
        except Exception as e:
            logger.error(f"Inpainting failed with {method}: {e}")
            # Fallback to traditional method
            fallback_inpainter = self.inpainters['traditional_telea']
            return fallback_inpainter.inpaint(frame, mask)
    
    def inpaint_face_region(self, 
                           frame: Frame, 
                           face: Face, 
                           method: str = None,
                           mask_type: str = 'smart_detection',
                           **kwargs) -> Tuple[Frame, np.ndarray]:
        """
        Inpaint face region with specified method.
        
        Args:
            frame: Input frame
            face: Face object with detection info
            method: Inpainting method to use
            mask_type: Type of mask ('boundary', 'occlusion', 'smart_detection', 'custom')
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (inpainted_frame, mask_used)
        """
        if method is None:
            method = self.default_method
        
        if method not in self.inpainters:
            logger.warning(f"Unknown inpainting method: {method}, using default")
            method = self.default_method
        
        inpainter = self.inpainters[method]
        if not inpainter.is_available():
            logger.warning(f"Inpainting method {method} not available, using fallback")
            inpainter = self.inpainters['traditional_telea']
        
        # Generate appropriate mask
        if mask_type == 'smart_detection':
            # Use advanced obstruction detection
            if hasattr(face, 'bbox'):
                face_bbox = tuple(map(int, face.bbox))
                mask = self.obstruction_detector.detect_all_obstructions(frame, face_bbox)
            else:
                # Fallback to boundary mask
                mask = InpaintingMaskGenerator.create_face_boundary_mask(
                    face, frame.shape, kwargs.get('expand_ratio', 0.1)
                )
        elif mask_type == 'boundary':
            mask = InpaintingMaskGenerator.create_face_boundary_mask(
                face, frame.shape, kwargs.get('expand_ratio', 0.1)
            )
        elif mask_type == 'occlusion':
            mask = InpaintingMaskGenerator.create_occlusion_mask(
                frame, kwargs.get('text_prompt')
            )
        elif mask_type == 'custom' and 'mask' in kwargs:
            mask = kwargs['mask']
        else:
            # Default to smart detection
            if hasattr(face, 'bbox'):
                face_bbox = tuple(map(int, face.bbox))
                mask = self.obstruction_detector.detect_all_obstructions(frame, face_bbox)
            else:
                mask = InpaintingMaskGenerator.create_face_boundary_mask(face, frame.shape)
        
        # Perform inpainting
        try:
            inpainted_frame = inpainter.inpaint(frame, mask, **kwargs)
            return inpainted_frame, mask
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return frame, mask  # Return original frame if inpainting fails
    
    def enhance_face_boundaries(self, 
                               swapped_frame: Frame,
                               original_frame: Frame,
                               face: Face,
                               blend_strength: float = 0.7) -> Frame:
        """
        Enhance face boundaries after swapping using inpainting.
        
        Args:
            swapped_frame: Frame with face already swapped
            original_frame: Original frame before swapping
            face: Face detection info
            blend_strength: Strength of the blending (0.0 - 1.0)
            
        Returns:
            Enhanced frame with improved boundaries
        """
        # Create soft boundary mask for blending
        mask = InpaintingMaskGenerator.create_face_boundary_mask(
            face, swapped_frame.shape, expand_ratio=0.05
        )
        
        # Apply Gaussian blur to mask for soft blending
        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
        mask_normalized = mask_blurred.astype(np.float32) / 255.0
        
        # Use traditional inpainting on boundary regions
        boundary_mask = cv2.Canny(mask, 50, 150)
        boundary_mask = cv2.dilate(boundary_mask, np.ones((3, 3), np.uint8))
        
        if np.any(boundary_mask > 0):
            inpainter = self.inpainters['traditional_telea']
            enhanced_frame = inpainter.inpaint(swapped_frame, boundary_mask)
            
            # Blend enhanced boundaries with original swap
            for c in range(3):
                enhanced_frame[:, :, c] = (
                    swapped_frame[:, :, c] * (1 - mask_normalized * blend_strength) +
                    enhanced_frame[:, :, c] * mask_normalized * blend_strength
                )
            
            return enhanced_frame.astype(np.uint8)
        
        return swapped_frame


# Global inpainting manager instance
INPAINTING_MANAGER = None


def get_inpainting_manager() -> InpaintingManager:
    """Get global inpainting manager instance."""
    global INPAINTING_MANAGER
    if INPAINTING_MANAGER is None:
        INPAINTING_MANAGER = InpaintingManager()
    return INPAINTING_MANAGER


def get_available_inpainting_methods() -> List[str]:
    """Get list of available inpainting methods."""
    return get_inpainting_manager().get_available_methods()


def inpaint_face_region(frame: Frame, face: Face, method: str = None, **kwargs) -> Tuple[Frame, np.ndarray]:
    """Convenience function for face region inpainting."""
    return get_inpainting_manager().inpaint_face_region(frame, face, method, **kwargs)


def enhance_face_boundaries(swapped_frame: Frame, original_frame: Frame, 
                          face: Face, blend_strength: float = 0.7) -> Frame:
    """Convenience function for face boundary enhancement."""
    return get_inpainting_manager().enhance_face_boundaries(
        swapped_frame, original_frame, face, blend_strength
    )