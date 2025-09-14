"""
Temporal consistency module for video face swapping.
Implements ComfyUI-inspired techniques for smooth video transitions.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from collections import deque
from dataclasses import dataclass

from roop.typing import Frame, Face
from roop.utilities import compute_cosine_distance

logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Information about a processed frame."""
    frame_number: int
    frame: Frame
    faces: List[Face]
    face_embeddings: List[np.ndarray]
    quality_scores: List[float]
    processing_time: float


class TemporalBuffer:
    """Buffer for managing temporal information across frames."""
    
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.frames: deque = deque(maxlen=buffer_size)
        self.face_tracks: Dict[int, List[Face]] = {}
        self.current_frame_number = 0
    
    def add_frame(self, frame_info: FrameInfo):
        """Add a new frame to the buffer."""
        self.frames.append(frame_info)
        self.current_frame_number = frame_info.frame_number
        
        # Update face tracks
        self._update_face_tracks(frame_info)
    
    def _update_face_tracks(self, frame_info: FrameInfo):
        """Update face tracking information."""
        if not frame_info.faces:
            return
        
        # Simple face tracking based on position and similarity
        for face in frame_info.faces:
            track_id = self._find_matching_track(face, frame_info.face_embeddings)
            if track_id is None:
                track_id = len(self.face_tracks)
                self.face_tracks[track_id] = []
            
            self.face_tracks[track_id].append(face)
            
            # Keep only recent faces in track
            if len(self.face_tracks[track_id]) > self.buffer_size:
                self.face_tracks[track_id].pop(0)
    
    def _find_matching_track(self, face: Face, embeddings: List[np.ndarray]) -> Optional[int]:
        """Find matching face track based on similarity."""
        if not hasattr(face, 'bbox') or not embeddings:
            return None
        
        best_track = None
        best_score = float('inf')
        
        for track_id, track_faces in self.face_tracks.items():
            if not track_faces:
                continue
            
            recent_face = track_faces[-1]
            if not hasattr(recent_face, 'bbox'):
                continue
            
            # Calculate position distance
            pos_dist = self._calculate_position_distance(face.bbox, recent_face.bbox)
            
            # Calculate embedding distance if available
            embed_dist = 0.0
            if (hasattr(face, 'embedding') and hasattr(recent_face, 'embedding') and
                face.embedding is not None and recent_face.embedding is not None):
                embed_dist = compute_cosine_distance(face.embedding, recent_face.embedding)
            
            # Combined score (lower is better)
            combined_score = pos_dist + embed_dist * 2.0
            
            if combined_score < best_score and combined_score < 0.5:  # Threshold
                best_score = combined_score
                best_track = track_id
        
        return best_track
    
    def _calculate_position_distance(self, bbox1, bbox2) -> float:
        """Calculate normalized position distance between bounding boxes."""
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        
        # Normalize by bbox size
        size1 = max(bbox1[2] - bbox1[0], bbox1[3] - bbox1[1])
        size2 = max(bbox2[2] - bbox2[0], bbox2[3] - bbox2[1])
        avg_size = (size1 + size2) / 2
        
        distance = np.linalg.norm(center1 - center2)
        return distance / max(avg_size, 1.0)
    
    def get_recent_frames(self, count: int = None) -> List[FrameInfo]:
        """Get recent frames from buffer."""
        if count is None:
            count = len(self.frames)
        return list(self.frames)[-count:]
    
    def get_face_trajectory(self, track_id: int) -> List[Face]:
        """Get face trajectory for a specific track."""
        return self.face_tracks.get(track_id, [])


class TemporalStabilizer:
    """Stabilizes face positions and features across frames."""
    
    def __init__(self, smoothing_factor: float = 0.3):
        self.smoothing_factor = smoothing_factor
        self.previous_positions = {}
        self.previous_landmarks = {}
    
    def stabilize_face_position(self, face: Face, track_id: int) -> Face:
        """Stabilize face position using temporal smoothing."""
        if not hasattr(face, 'bbox'):
            return face
        
        current_bbox = np.array(face.bbox)
        
        if track_id in self.previous_positions:
            previous_bbox = self.previous_positions[track_id]
            # Apply exponential smoothing
            smoothed_bbox = (self.smoothing_factor * current_bbox + 
                           (1 - self.smoothing_factor) * previous_bbox)
            face.bbox = smoothed_bbox.tolist()
        
        self.previous_positions[track_id] = current_bbox
        return face
    
    def stabilize_landmarks(self, face: Face, track_id: int) -> Face:
        """Stabilize facial landmarks using temporal smoothing."""
        if not hasattr(face, 'kps') or face.kps is None:
            return face
        
        current_landmarks = np.array(face.kps)
        
        if track_id in self.previous_landmarks:
            previous_landmarks = self.previous_landmarks[track_id]
            # Apply exponential smoothing
            smoothed_landmarks = (self.smoothing_factor * current_landmarks +
                                (1 - self.smoothing_factor) * previous_landmarks)
            face.kps = smoothed_landmarks.tolist()
        
        self.previous_landmarks[track_id] = current_landmarks
        return face


class OpticalFlowTracker:
    """Tracks faces using optical flow for improved temporal consistency."""
    
    def __init__(self):
        self.previous_gray = None
        self.tracked_points = {}
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
    
    def track_faces(self, current_frame: Frame, faces: List[Face]) -> List[Face]:
        """Track faces using optical flow."""
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if self.previous_gray is None:
            self.previous_gray = current_gray
            return faces
        
        tracked_faces = []
        
        for i, face in enumerate(faces):
            if hasattr(face, 'kps') and face.kps is not None:
                # Track facial landmarks
                landmarks = np.array(face.kps, dtype=np.float32).reshape(-1, 1, 2)
                
                # Calculate optical flow
                new_landmarks, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.previous_gray, current_gray, landmarks, None, **self.lk_params
                )
                
                # Filter valid points
                valid_landmarks = new_landmarks[status.flatten() == 1]
                
                if len(valid_landmarks) >= len(landmarks) * 0.5:  # At least 50% tracked
                    face.kps = new_landmarks.reshape(-1, 2).tolist()
                
            tracked_faces.append(face)
        
        self.previous_gray = current_gray
        return tracked_faces


class FrameInterpolator:
    """Interpolates between frames for smoother transitions."""
    
    @staticmethod
    def interpolate_face_features(face1: Face, face2: Face, alpha: float) -> Face:
        """Interpolate between two faces."""
        interpolated_face = Face()
        
        # Interpolate bounding box
        if hasattr(face1, 'bbox') and hasattr(face2, 'bbox'):
            bbox1 = np.array(face1.bbox)
            bbox2 = np.array(face2.bbox)
            interpolated_face.bbox = ((1 - alpha) * bbox1 + alpha * bbox2).tolist()
        
        # Interpolate landmarks
        if (hasattr(face1, 'kps') and hasattr(face2, 'kps') and
            face1.kps is not None and face2.kps is not None):
            kps1 = np.array(face1.kps)
            kps2 = np.array(face2.kps)
            interpolated_face.kps = ((1 - alpha) * kps1 + alpha * kps2).tolist()
        
        # Interpolate embedding if possible
        if (hasattr(face1, 'embedding') and hasattr(face2, 'embedding') and
            face1.embedding is not None and face2.embedding is not None):
            embed1 = np.array(face1.embedding)
            embed2 = np.array(face2.embedding)
            interpolated_face.embedding = ((1 - alpha) * embed1 + alpha * embed2)
        
        return interpolated_face
    
    @staticmethod
    def create_transition_frame(frame1: Frame, frame2: Frame, alpha: float) -> Frame:
        """Create interpolated frame between two frames."""
        return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)


class TemporalConsistencyManager:
    """Main manager for temporal consistency in video processing."""
    
    def __init__(self, buffer_size: int = 5, smoothing_factor: float = 0.3):
        self.buffer = TemporalBuffer(buffer_size)
        self.stabilizer = TemporalStabilizer(smoothing_factor)
        self.optical_flow_tracker = OpticalFlowTracker()
        self.frame_interpolator = FrameInterpolator()
        
        self.enable_position_stabilization = True
        self.enable_landmark_stabilization = True
        self.enable_optical_flow = True
        self.enable_quality_filtering = True
        
        self.quality_threshold = 0.5
    
    def process_frame(self, frame: Frame, faces: List[Face], 
                     frame_number: int) -> Tuple[List[Face], Dict[str, Any]]:
        """
        Process frame with temporal consistency.
        
        Args:
            frame: Input frame
            faces: Detected faces
            frame_number: Frame number in sequence
            
        Returns:
            Tuple of (processed_faces, metadata)
        """
        processing_start = cv2.getTickCount()
        
        # Track faces using optical flow if enabled
        if self.enable_optical_flow and faces:
            faces = self.optical_flow_tracker.track_faces(frame, faces)
        
        # Stabilize face positions and landmarks
        stabilized_faces = []
        face_embeddings = []
        quality_scores = []
        
        for i, face in enumerate(faces):
            track_id = i  # Simplified tracking ID
            
            # Position stabilization
            if self.enable_position_stabilization:
                face = self.stabilizer.stabilize_face_position(face, track_id)
            
            # Landmark stabilization
            if self.enable_landmark_stabilization:
                face = self.stabilizer.stabilize_landmarks(face, track_id)
            
            # Calculate quality score
            quality_score = self._calculate_face_quality(face, frame)
            quality_scores.append(quality_score)
            
            # Extract embedding if available
            embedding = getattr(face, 'embedding', None)
            face_embeddings.append(embedding)
            
            stabilized_faces.append(face)
        
        # Filter faces by quality if enabled
        if self.enable_quality_filtering:
            filtered_faces = []
            for face, quality in zip(stabilized_faces, quality_scores):
                if quality >= self.quality_threshold:
                    filtered_faces.append(face)
            stabilized_faces = filtered_faces
        
        # Calculate processing time
        processing_time = (cv2.getTickCount() - processing_start) / cv2.getTickFrequency()
        
        # Create frame info and add to buffer
        frame_info = FrameInfo(
            frame_number=frame_number,
            frame=frame,
            faces=stabilized_faces,
            face_embeddings=face_embeddings,
            quality_scores=quality_scores,
            processing_time=processing_time
        )
        
        self.buffer.add_frame(frame_info)
        
        # Generate metadata
        metadata = {
            'frame_number': frame_number,
            'faces_detected': len(faces),
            'faces_after_filtering': len(stabilized_faces),
            'average_quality': np.mean(quality_scores) if quality_scores else 0.0,
            'processing_time': processing_time,
            'buffer_size': len(self.buffer.frames),
            'face_tracks': len(self.buffer.face_tracks)
        }
        
        return stabilized_faces, metadata
    
    def _calculate_face_quality(self, face: Face, frame: Frame) -> float:
        """Calculate quality score for a face."""
        quality_score = 0.5  # Default
        
        try:
            if hasattr(face, 'bbox'):
                x1, y1, x2, y2 = map(int, face.bbox)
                face_region = frame[y1:y2, x1:x2]
                
                if face_region.size > 0:
                    # Calculate sharpness using Laplacian variance
                    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    # Normalize sharpness score
                    sharpness_score = min(laplacian_var / 1000.0, 1.0)
                    
                    # Calculate size score (larger faces generally better)
                    face_area = (x2 - x1) * (y2 - y1)
                    size_score = min(face_area / (100 * 100), 1.0)  # Normalize to 100x100
                    
                    # Combined quality score
                    quality_score = (sharpness_score * 0.7 + size_score * 0.3)
        
        except Exception as e:
            logger.warning(f"Error calculating face quality: {e}")
        
        return quality_score
    
    def get_temporal_info(self) -> Dict[str, Any]:
        """Get temporal processing information."""
        recent_frames = self.buffer.get_recent_frames(3)
        
        info = {
            'total_frames_processed': self.buffer.current_frame_number + 1,
            'buffer_size': len(self.buffer.frames),
            'active_face_tracks': len(self.buffer.face_tracks),
            'average_faces_per_frame': 0.0,
            'average_quality': 0.0,
            'average_processing_time': 0.0
        }
        
        if recent_frames:
            total_faces = sum(len(frame.faces) for frame in recent_frames)
            total_quality = sum(sum(frame.quality_scores) for frame in recent_frames if frame.quality_scores)
            total_quality_count = sum(len(frame.quality_scores) for frame in recent_frames)
            total_time = sum(frame.processing_time for frame in recent_frames)
            
            info['average_faces_per_frame'] = total_faces / len(recent_frames)
            info['average_quality'] = total_quality / max(total_quality_count, 1)
            info['average_processing_time'] = total_time / len(recent_frames)
        
        return info
    
    def reset(self):
        """Reset temporal state for new video."""
        self.buffer = TemporalBuffer(self.buffer.buffer_size)
        self.stabilizer = TemporalStabilizer(self.stabilizer.smoothing_factor)
        self.optical_flow_tracker = OpticalFlowTracker()


# Global temporal consistency manager
TEMPORAL_MANAGER = None


def get_temporal_manager() -> TemporalConsistencyManager:
    """Get global temporal consistency manager."""
    global TEMPORAL_MANAGER
    if TEMPORAL_MANAGER is None:
        TEMPORAL_MANAGER = TemporalConsistencyManager()
    return TEMPORAL_MANAGER


def process_frame_with_temporal_consistency(frame: Frame, faces: List[Face], 
                                          frame_number: int) -> Tuple[List[Face], Dict[str, Any]]:
    """Convenience function for temporal processing."""
    return get_temporal_manager().process_frame(frame, faces, frame_number)


def reset_temporal_state():
    """Reset temporal state for new video."""
    manager = get_temporal_manager()
    if manager is not None:
        manager.reset()


def get_temporal_info() -> Dict[str, Any]:
    """Get temporal processing information."""
    return get_temporal_manager().get_temporal_info()