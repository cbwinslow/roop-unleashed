from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_util import get_first_face, get_all_faces
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video, compute_cosine_distance, get_destfilename_from_path

# Import enhanced face processing capabilities
try:
    from roop.enhanced_face_swapper import (
        enhanced_process_frame, enhanced_process_frames, get_enhancement_config,
        assess_frame_quality, get_available_blend_methods
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Enhanced face processing not available - using standard processing")

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

DIST_THRESHOLD = 0.65


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.')
        return False
    elif not get_first_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.')
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.')
        return False
    return True


def post_process() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    global DIST_THRESHOLD

    # Check if enhanced processing is available and enabled
    if ENHANCED_AVAILABLE and hasattr(roop.globals, 'use_enhanced_processing') and roop.globals.use_enhanced_processing:
        # Use enhanced processing with quality assessment
        blend_method = getattr(roop.globals, 'blend_method', 'multiband')
        blend_ratio = getattr(roop.globals, 'blend_ratio', 0.8)
        
        selection_mode = "all_faces" if roop.globals.many_faces else "match_target"
        
        return enhanced_process_frame(
            source_face, target_face, temp_frame,
            face_selection_mode=selection_mode,
            blend_method=blend_method,
            blend_ratio=blend_ratio
        )
    
    # Standard processing (existing code)
    if roop.globals.many_faces:
        many_faces = get_all_faces(temp_frame)
        if many_faces is not None:
            for target_face in many_faces:
                if target_face['det_score'] > 0.65:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        if target_face:
            target_embedding = target_face.embedding
            many_faces = get_all_faces(temp_frame)
            target_face = None
            for dest_face in many_faces:
                dest_embedding = dest_face.embedding
                if compute_cosine_distance(target_embedding, dest_embedding) <= DIST_THRESHOLD:
                    target_face = dest_face
                    break
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)
            return temp_frame
                    
        target_face = get_first_face(temp_frame)
        if target_face is not None:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame



def process_frames(is_batch: bool, source_face: Face, target_face: Face, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    # Check if enhanced processing is available and enabled
    if ENHANCED_AVAILABLE and hasattr(roop.globals, 'use_enhanced_processing') and roop.globals.use_enhanced_processing:
        blend_method = getattr(roop.globals, 'blend_method', 'multiband')
        blend_ratio = getattr(roop.globals, 'blend_ratio', 0.8)
        
        enhanced_process_frames(is_batch, source_face, target_face, temp_frame_paths, update, blend_method, blend_ratio)
        return
    
    # Standard processing (existing code)
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is not None:
            result = process_frame(source_face, target_face, temp_frame)
            if result is not None:
                if is_batch:
                    tf = get_destfilename_from_path(temp_frame_path, roop.globals.output_path, '_fake.png')
                    cv2.imwrite(tf, result)
                else:
                    cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_face: Any, target_face: Any, target_path: str, output_path: str) -> None:
    global DIST_THRESHOLD

    target_frame = cv2.imread(target_path)
    if target_frame is not None:
        result = process_frame(source_face, target_face, target_frame)
        if result is not None:
            cv2.imwrite(output_path, result)


def process_video(source_face: Any, target_face: Any, temp_frame_paths: List[str]) -> None:
    global DIST_THRESHOLD

    roop.processors.frame.core.process_video(source_face, target_face, temp_frame_paths, process_frames)


def process_batch_images(source_face: Any, target_face: Any, temp_frame_paths: List[str]) -> None:
    global DIST_THRESHOLD

    roop.processors.frame.core.process_batch(source_face, target_face, temp_frame_paths, process_frames)


# Enhanced processing utilities
def get_processing_info() -> dict:
    """Get information about available processing methods."""
    info = {
        "enhanced_available": ENHANCED_AVAILABLE,
        "standard_processing": True
    }
    
    if ENHANCED_AVAILABLE:
        info.update(get_enhancement_config())
    
    return info


def assess_image_quality(image_path: str) -> dict:
    """Assess the quality of an image for face swapping."""
    if not ENHANCED_AVAILABLE:
        return {"error": "Enhanced processing not available"}
    
    frame = cv2.imread(image_path)
    if frame is None:
        return {"error": "Could not load image"}
    
    return assess_frame_quality(frame)


def get_blend_methods() -> List[str]:
    """Get available blending methods."""
    if ENHANCED_AVAILABLE:
        return get_available_blend_methods()
    return ["standard"]


def enable_enhanced_processing(enable: bool = True) -> None:
    """Enable or disable enhanced processing."""
    if ENHANCED_AVAILABLE:
        roop.globals.use_enhanced_processing = enable
        print(f"Enhanced processing {'enabled' if enable else 'disabled'}")
    else:
        print("Enhanced processing not available")


def set_blend_method(method: str) -> bool:
    """Set the blending method for enhanced processing."""
    if not ENHANCED_AVAILABLE:
        return False
    
    available_methods = get_available_blend_methods()
    if method in available_methods:
        roop.globals.blend_method = method
        print(f"Blend method set to: {method}")
        return True
    else:
        print(f"Invalid blend method. Available: {available_methods}")
        return False
