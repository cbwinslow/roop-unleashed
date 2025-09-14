"""
Mock insightface module for testing and development when insightface is not available.
This provides basic functionality to allow the enhanced modules to be tested.
"""

import cv2
import numpy as np
from typing import List, Optional, Any, Tuple


class MockFace:
    """Mock Face object that mimics insightface Face structure."""
    
    def __init__(self):
        self.bbox = np.array([0, 0, 100, 100])
        self.kps = np.array([[20, 30], [80, 30], [50, 50], [30, 70], [70, 70]])  # Basic 5-point landmarks
        self.landmark_2d_106 = None  # Will be set if needed
        self.det_score = 0.9
        self.embedding = None
    
    def set_from_detection(self, bbox: np.ndarray, landmarks: np.ndarray, score: float = 0.9):
        """Set face properties from detection results."""
        self.bbox = bbox
        self.kps = landmarks
        self.det_score = score
        
        # Generate fake 106-point landmarks if needed
        if landmarks is not None and len(landmarks) >= 5:
            self.landmark_2d_106 = self._generate_106_landmarks(landmarks)
    
    def _generate_106_landmarks(self, kps: np.ndarray) -> np.ndarray:
        """Generate mock 106-point landmarks from 5-point landmarks."""
        # This is a simplified approximation
        landmarks_106 = np.zeros((106, 2))
        
        if len(kps) >= 5:
            # Use the 5-point landmarks as base
            left_eye = kps[0]
            right_eye = kps[1] 
            nose = kps[2]
            left_mouth = kps[3]
            right_mouth = kps[4]
            
            # Face contour (points 0-32)
            face_width = abs(right_eye[0] - left_eye[0]) * 1.5
            face_center = (left_eye + right_eye) / 2
            
            for i in range(33):
                angle = (i / 32) * np.pi - np.pi/2
                x = face_center[0] + face_width * np.cos(angle) * 0.8
                y = face_center[1] + face_width * np.sin(angle) * 1.1
                landmarks_106[i] = [x, y]
            
            # Left eyebrow (points 33-41)
            for i in range(9):
                x = left_eye[0] + (i - 4) * 3
                y = left_eye[1] - 10
                landmarks_106[33 + i] = [x, y]
            
            # Right eyebrow (points 42-50)
            for i in range(9):
                x = right_eye[0] + (i - 4) * 3
                y = right_eye[1] - 10
                landmarks_106[42 + i] = [x, y]
            
            # Nose (points 51-85)
            for i in range(35):
                x = nose[0] + (i - 17) * 1.5
                y = nose[1] + (i % 7 - 3) * 2
                landmarks_106[51 + i] = [x, y]
            
            # Mouth (points 86-105)
            mouth_center = (left_mouth + right_mouth) / 2
            mouth_width = abs(right_mouth[0] - left_mouth[0])
            
            for i in range(20):
                angle = (i / 19) * 2 * np.pi
                x = mouth_center[0] + (mouth_width / 2) * np.cos(angle)
                y = mouth_center[1] + (mouth_width / 4) * np.sin(angle)
                landmarks_106[86 + i] = [x, y]
        
        return landmarks_106


class MockFaceAnalysis:
    """Mock FaceAnalysis that mimics insightface.app.FaceAnalysis."""
    
    def __init__(self, name: str = 'buffalo_l', providers: List[str] = None):
        self.name = name
        self.providers = providers or ['CPUExecutionProvider']
        self.det_size = (640, 640)
        self.is_prepared = False
    
    def prepare(self, ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)):
        """Prepare the face analysis model."""
        self.det_size = det_size
        self.is_prepared = True
        print(f"Mock FaceAnalysis prepared with det_size={det_size}")
    
    def get(self, image: np.ndarray, max_num: int = 0) -> List[MockFace]:
        """Mock face detection that returns fake faces."""
        if not self.is_prepared:
            self.prepare()
        
        # Simple face detection using OpenCV Haar cascades as fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Try to load Haar cascade for face detection
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            detected_faces = []
            for (x, y, w, h) in faces:
                face = MockFace()
                
                # Set bounding box
                bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
                
                # Generate basic 5-point landmarks
                landmarks = np.array([
                    [x + w * 0.3, y + h * 0.35],  # Left eye
                    [x + w * 0.7, y + h * 0.35],  # Right eye  
                    [x + w * 0.5, y + h * 0.55],  # Nose
                    [x + w * 0.35, y + h * 0.75], # Left mouth corner
                    [x + w * 0.65, y + h * 0.75]  # Right mouth corner
                ], dtype=np.float32)
                
                face.set_from_detection(bbox, landmarks, 0.8)
                
                # Generate a simple embedding (random but consistent)
                np.random.seed(int(x + y + w + h) % 1000)
                face.embedding = np.random.rand(512).astype(np.float32)
                
                detected_faces.append(face)
                
                if max_num > 0 and len(detected_faces) >= max_num:
                    break
            
            return detected_faces
            
        except Exception as e:
            print(f"Mock face detection failed: {e}")
            return []


class MockApp:
    """Mock app module that mimics insightface.app."""
    
    FaceAnalysis = MockFaceAnalysis


# Create mock insightface module structure
class MockInsightFace:
    """Mock insightface module."""
    
    def __init__(self):
        self.app = MockApp()


# Global mock instance
_mock_insightface = MockInsightFace()

# Export the main classes and functions
Face = MockFace
app = _mock_insightface.app