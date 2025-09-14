"""
Advanced blending techniques for seamless face integration.
Implements Poisson blending, multi-band blending, and gradient-based methods.
"""

import cv2
import numpy as np
from typing import Tuple
from scipy.sparse.linalg import spsolve


class PoissonBlending:
    """Implementation of Poisson blending for seamless image integration."""
    
    @staticmethod
    def poisson_blend(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform Poisson blending to seamlessly integrate source into target.
        
        Args:
            source: Source image (face to be blended)
            target: Target image (background)
            mask: Binary mask defining the blending region
            
        Returns:
            Blended image
        """
        if source.shape != target.shape:
            source = cv2.resize(source, (target.shape[1], target.shape[0]))
        
        if mask.shape[:2] != target.shape[:2]:
            mask = cv2.resize(mask, (target.shape[1], target.shape[0]))
        
        # Ensure mask is binary
        mask = (mask > 128).astype(np.uint8)
        
        result = target.copy()
        
        for channel in range(3):
            # Solve Poisson equation for each color channel
            blended_channel = PoissonBlending._solve_poisson(
                source[:, :, channel], 
                target[:, :, channel], 
                mask
            )
            result[:, :, channel] = blended_channel
        
        return result
    
    @staticmethod
    def _solve_poisson(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Solve Poisson equation for one channel."""
        h, w = source.shape
        
        # Create index mapping
        indices = np.zeros((h, w), dtype=int)
        mask_indices = np.where(mask > 0)
        num_variables = len(mask_indices[0])
        
        if num_variables == 0:
            return target
        
        indices[mask_indices] = np.arange(num_variables)
        
        # Build coefficient matrix A and right-hand side b
        A_data, A_row, A_col = [], [], []
        b = np.zeros(num_variables)
        
        for idx in range(num_variables):
            i, j = mask_indices[0][idx], mask_indices[1][idx]
            var_idx = indices[i, j]
            
            # Laplacian coefficient for current pixel
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            valid_neighbors = [(ni, nj) for ni, nj in neighbors 
                             if 0 <= ni < h and 0 <= nj < w]
            
            A_data.append(len(valid_neighbors))
            A_row.append(var_idx)
            A_col.append(var_idx)
            
            # Guidance field (Laplacian of source)
            guidance = 0
            boundary_sum = 0
            
            for ni, nj in valid_neighbors:
                guidance += source[i, j] - source[ni, nj]
                
                if mask[ni, nj] == 0:  # Boundary condition
                    boundary_sum += target[ni, nj]
                else:  # Interior point
                    neighbor_idx = indices[ni, nj]
                    A_data.append(-1)
                    A_row.append(var_idx)
                    A_col.append(neighbor_idx)
            
            b[var_idx] = guidance + boundary_sum
        
        # Solve sparse linear system
        from scipy.sparse import csr_matrix
        A = csr_matrix((A_data, (A_row, A_col)), shape=(num_variables, num_variables))
        solution = spsolve(A, b)
        
        # Reconstruct image
        result = target.copy()
        for idx in range(num_variables):
            i, j = mask_indices[0][idx], mask_indices[1][idx]
            result[i, j] = np.clip(solution[idx], 0, 255)
        
        return result


class MultiBandBlending:
    """Multi-band blending for improved color and texture matching."""
    
    @staticmethod
    def multiband_blend(source: np.ndarray, target: np.ndarray, mask: np.ndarray, 
                       levels: int = 4) -> np.ndarray:
        """
        Perform multi-band blending using Laplacian pyramids.
        
        Args:
            source: Source image
            target: Target image  
            mask: Blending mask
            levels: Number of pyramid levels
            
        Returns:
            Blended image
        """
        if source.shape != target.shape:
            source = cv2.resize(source, (target.shape[1], target.shape[0]))
        
        if mask.shape[:2] != target.shape[:2]:
            mask = cv2.resize(mask, (target.shape[1], target.shape[0]))
        
        # Normalize mask to [0, 1]
        mask_float = mask.astype(np.float32) / 255.0
        if len(mask_float.shape) == 2:
            mask_float = np.stack([mask_float] * 3, axis=2)
        
        # Build Gaussian pyramid for mask
        mask_pyramid = MultiBandBlending._build_gaussian_pyramid(mask_float, levels)
        
        # Build Laplacian pyramids for source and target
        source_pyramid = MultiBandBlending._build_laplacian_pyramid(source.astype(np.float32), levels)
        target_pyramid = MultiBandBlending._build_laplacian_pyramid(target.astype(np.float32), levels)
        
        # Blend pyramids
        blended_pyramid = []
        for i in range(levels):
            blended_level = (source_pyramid[i] * mask_pyramid[i] + 
                           target_pyramid[i] * (1 - mask_pyramid[i]))
            blended_pyramid.append(blended_level)
        
        # Reconstruct from pyramid
        result = MultiBandBlending._reconstruct_from_pyramid(blended_pyramid)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _build_gaussian_pyramid(image: np.ndarray, levels: int) -> list:
        """Build Gaussian pyramid."""
        pyramid = [image]
        current = image
        
        for _ in range(levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        return pyramid
    
    @staticmethod
    def _build_laplacian_pyramid(image: np.ndarray, levels: int) -> list:
        """Build Laplacian pyramid."""
        gaussian_pyramid = MultiBandBlending._build_gaussian_pyramid(image, levels)
        laplacian_pyramid = []
        
        for i in range(levels - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = gaussian_pyramid[i] - upsampled
            laplacian_pyramid.append(laplacian)
        
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid
    
    @staticmethod
    def _reconstruct_from_pyramid(pyramid: list) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        current = pyramid[-1]
        
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            current = cv2.pyrUp(current, dstsize=size)
            current = current + pyramid[i]
        
        return current


class GradientBlending:
    """Gradient-based blending techniques."""
    
    @staticmethod
    def gradient_blend(source: np.ndarray, target: np.ndarray, mask: np.ndarray, 
                      alpha: float = 0.8) -> np.ndarray:
        """
        Perform gradient-based blending with edge preservation.
        
        Args:
            source: Source image
            target: Target image
            mask: Blending mask
            alpha: Blending strength (0.0 to 1.0)
            
        Returns:
            Blended image
        """
        if source.shape != target.shape:
            source = cv2.resize(source, (target.shape[1], target.shape[0]))
        
        if mask.shape[:2] != target.shape[:2]:
            mask = cv2.resize(mask, (target.shape[1], target.shape[0]))
        
        # Calculate gradients
        source_grad = GradientBlending._calculate_gradients(source)
        target_grad = GradientBlending._calculate_gradients(target)
        
        # Blend gradients
        mask_float = mask.astype(np.float32) / 255.0
        if len(mask_float.shape) == 2:
            mask_float = np.stack([mask_float] * 3, axis=2)
        
        blended_grad = (source_grad * mask_float * alpha + 
                       target_grad * (1 - mask_float * alpha))
        
        # Integrate gradients to get final image
        result = GradientBlending._integrate_gradients(blended_grad, target)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _calculate_gradients(image: np.ndarray) -> np.ndarray:
        """Calculate image gradients using Sobel operators."""
        if len(image.shape) == 3:
            gradients = np.zeros_like(image, dtype=np.float32)
            for i in range(3):
                grad_x = cv2.Sobel(image[:, :, i], cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(image[:, :, i], cv2.CV_32F, 0, 1, ksize=3)
                gradients[:, :, i] = np.sqrt(grad_x**2 + grad_y**2)
            return gradients
        else:
            grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            return np.sqrt(grad_x**2 + grad_y**2)
    
    @staticmethod
    def _integrate_gradients(gradients: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Integrate gradients back to image using reference."""
        # Simple integration using reference as base
        return reference + (gradients - GradientBlending._calculate_gradients(reference)) * 0.1


class AdvancedBlender:
    """Main class for advanced blending operations."""
    
    def __init__(self):
        self.poisson_blender = PoissonBlending()
        self.multiband_blender = MultiBandBlending()
        self.gradient_blender = GradientBlending()
    
    def blend_face(self, source_face: np.ndarray, target_frame: np.ndarray, 
                   face_bbox: Tuple[int, int, int, int], 
                   blend_method: str = "multiband", 
                   blend_ratio: float = 0.8) -> np.ndarray:
        """
        Blend source face into target frame using specified method.
        
        Args:
            source_face: Face image to blend
            target_frame: Target frame
            face_bbox: Face bounding box (x1, y1, x2, y2)
            blend_method: Blending method ("poisson", "multiband", "gradient", "alpha")
            blend_ratio: Blending strength
            
        Returns:
            Blended frame
        """
        x1, y1, x2, y2 = [int(coord) for coord in face_bbox]
        
        # Extract target face region
        target_face = target_frame[y1:y2, x1:x2].copy()
        
        # Resize source face to match target
        if source_face.shape[:2] != target_face.shape[:2]:
            source_face = cv2.resize(source_face, (target_face.shape[1], target_face.shape[0]))
        
        # Create mask for face region
        mask = self._create_face_mask(target_face.shape[:2])
        
        # Apply selected blending method
        if blend_method == "poisson":
            blended_face = self.poisson_blender.poisson_blend(source_face, target_face, mask)
        elif blend_method == "multiband":
            blended_face = self.multiband_blender.multiband_blend(source_face, target_face, mask)
        elif blend_method == "gradient":
            blended_face = self.gradient_blender.gradient_blend(source_face, target_face, mask, blend_ratio)
        else:  # alpha blending
            mask_float = mask.astype(np.float32) / 255.0
            if len(mask_float.shape) == 2:
                mask_float = np.stack([mask_float] * 3, axis=2)
            blended_face = (source_face * mask_float * blend_ratio + 
                          target_face * (1 - mask_float * blend_ratio))
            blended_face = np.clip(blended_face, 0, 255).astype(np.uint8)
        
        # Apply edge smoothing
        blended_face = self._apply_edge_smoothing(blended_face, target_face, mask)
        
        # Insert blended face back into frame
        result = target_frame.copy()
        result[y1:y2, x1:x2] = blended_face
        
        return result
    
    def _create_face_mask(self, face_shape: Tuple[int, int]) -> np.ndarray:
        """Create elliptical mask for face blending."""
        height, width = face_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create elliptical mask
        center_x, center_y = width // 2, height // 2
        axes_x, axes_y = int(width * 0.4), int(height * 0.45)
        
        cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
        
        # Apply Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def _apply_edge_smoothing(self, blended: np.ndarray, target: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
        """Apply edge smoothing to reduce artifacts."""
        # Create edge mask
        edge_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - cv2.erode(mask, np.ones((5, 5), np.uint8))
        edge_mask = cv2.GaussianBlur(edge_mask, (15, 15), 0)
        
        # Apply smoothing only at edges
        edge_factor = edge_mask.astype(np.float32) / 255.0
        if len(edge_factor.shape) == 2:
            edge_factor = np.stack([edge_factor] * 3, axis=2)
        
        smoothed = cv2.bilateralFilter(blended, 9, 75, 75)
        result = blended * (1 - edge_factor) + smoothed * edge_factor
        
        return np.clip(result, 0, 255).astype(np.uint8)


# Convenience functions
def advanced_blend_face(source_face: np.ndarray, target_frame: np.ndarray, 
                       face_bbox: Tuple[int, int, int, int], 
                       method: str = "multiband", ratio: float = 0.8) -> np.ndarray:
    """Convenience function for advanced face blending."""
    blender = AdvancedBlender()
    return blender.blend_face(source_face, target_frame, face_bbox, method, ratio)


def get_available_blend_methods() -> list:
    """Get list of available blending methods."""
    return ["alpha", "multiband", "gradient", "poisson"]