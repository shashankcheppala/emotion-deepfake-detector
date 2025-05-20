import numpy as np
import cv2
from pathlib import Path

def extract_frames_from_video(video_path, num_frames=8, img_size=(224, 224)):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return np.zeros((num_frames, *img_size, 3), dtype=np.float32)

    idxs = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    result = []
    for i in idxs:
        frame = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), img_size)
        result.append(frame.astype(np.float32) / 255.0)

    while len(result) < num_frames:
        result.append(result[-1].copy())
    return np.stack(result)
