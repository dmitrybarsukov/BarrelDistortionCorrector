import cv2
import numpy as np
from typing import Tuple

def process_frame(
        img: np.ndarray,
        barrel_coef: float,
        corners_coef: float) -> np.ndarray:

    if img is None:
        return img

    height, width = img.shape[0:2]
    dist_cft = np.zeros((4, 1), np.float64)

    dist_cft[0, 0] = barrel_coef
    dist_cft[1, 0] = corners_coef

    cam = np.eye(3, dtype=np.float32)

    cam[0, 2] = width * 0.5
    cam[1, 2] = height * 0.5
    cam[0, 0] = max(width, height)
    cam[1, 1] = max(width, height)
    return cv2.undistort(img, cam, dist_cft)


