from typing import Optional
import cv2

from roop.typing import Frame

def get_image_frame(filename: str):
    try:
        frame = cv2.imread(filename)
        if frame is None:
            print(f'Unable to read image {filename}')
        return frame
    except Exception as e:
        print(f'Error reading {filename}: {e}')
    return None



    
def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print(f'Unable to open video {video_path}')
            return None
        frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
        has_frame, frame = capture.read()
        capture.release()
        if has_frame:
            return frame
    except Exception as e:
        print(f'Error reading video {video_path}: {e}')
    return None


def get_video_frame_total(video_path: str) -> int:
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print(f'Unable to open video {video_path}')
            return 0
        video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return video_frame_total
    except Exception as e:
        print(f'Error getting frame count for {video_path}: {e}')
    return 0
