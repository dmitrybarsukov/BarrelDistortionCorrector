import cv2
import numpy as np
from typing import Tuple

class BaseDeFisheyer:
    def __init__(self,
                 width: int,
                 height: int,
                 depth: int,
                 border_size: float,
                 coeff2: float,
                 coeff4: float,
                 rotation: int = 0):
        # TODO image rotation
        # save parameters
        self.width = width
        self.height = height
        self.depth = depth
        self.border_size = border_size
        self.coeff2 = coeff2
        self.coeff4 = coeff4
        # calc internal values
        self.tmp_height = int(self.height * (1 + self.border_size * 2))
        self.tmp_width = int(self.width * (1 + self.border_size * 2))
        self.dh = (self.tmp_height - self.height) // 2
        self.dw = (self.tmp_width - self.width) // 2
        # create internal objects
        self.temp_img_in = np.zeros((self.tmp_height, self.tmp_width, self.depth), dtype=np.uint8)
        self.temp_img_out = np.zeros((self.tmp_height, self.tmp_width, self.depth), dtype=np.uint8)
        self.coef_matrix = np.zeros((4, 1), np.float64)
        self.coef_matrix[0, 0] = self.coeff2
        self.coef_matrix[1, 0] = self.coeff4
        self.cam_matrix = np.eye(3, dtype=np.float32)
        self.cam_matrix[0, 2] = self.tmp_width * 0.5
        self.cam_matrix[1, 2] = self.tmp_height * 0.5
        self.cam_matrix[0, 0] = max(self.tmp_width, self.tmp_height)
        self.cam_matrix[1, 1] = max(self.tmp_width, self.tmp_height)
        # calculate crop area afred defisheyeing
        self.temp_img_in[self.dh : self.dh + self.height, self.dw : self.dw + self.width] = 255
        self.__undistort_self()
        mid_vert = self.tmp_height // 2
        mid_hor = self.tmp_width // 2
        # horizontal crop zones
        self.left_crop = 0
        self.right_crop = self.tmp_width - 1
        for i in range(self.tmp_width - 1):
            if self.__is_black(self.temp_img_out[mid_vert, i]) and not self.__is_black(self.temp_img_out[mid_vert, i + 1]):
                self.left_crop = i + 2
            elif not self.__is_black(self.temp_img_out[mid_vert, i]) and self.__is_black(self.temp_img_out[mid_vert, i + 1]):
                self.right_crop = i
        self.top_crop = 0
        self.bottom_crop = self.tmp_height - 1
        # vertical crop zones
        for i in range(self.tmp_height - 1):
            if self.__is_black(self.temp_img_out[i, mid_hor]) and not self.__is_black(self.temp_img_out[i + 1, mid_hor]):
                self.top_crop = i + 2
            elif not self.__is_black(self.temp_img_out[i, mid_hor]) and self.__is_black(self.temp_img_out[i + 1, mid_hor]):
                self.bottom_crop = i

        # correct crop zones to match original aspect ratio
        orig_aspect = self.width / self.height
        cur_aspect = (self.right_crop - self.left_crop) / (self.bottom_crop - self.top_crop)
        while max(orig_aspect, cur_aspect) / min(orig_aspect, cur_aspect) > 1.01:
            if cur_aspect > orig_aspect:
                self.left_crop += 1
                self.right_crop -= 1
            else:
                self.top_crop += 1
                self.bottom_crop -= 1
            cur_aspect = (self.right_crop - self.left_crop) / (self.bottom_crop - self.top_crop)

    def __is_black(self, pixel) -> bool:
        if pixel is int and pixel == 0:
            return True
        return not any(pixel)

    def __check_frame_size(self, img: np.ndarray):
        if img.shape[0] != self.height or img.shape[1] != self.width:
            raise Exception(f"Invalid frame size. Expected {self.width}x{self.height}, got {img.shape[1]}x{img.shape[0]}")

    def __undistort_self(self):
        cv2.undistort(self.temp_img_in, self.cam_matrix, self.coef_matrix, dst=self.temp_img_out)

    def process_frame(self, img: np.ndarray) -> np.ndarray:
        self.__check_frame_size(img)
        self.temp_img_in[self.dh : self.dh + self.height, self.dw : self.dw + self.width] = img
        self.__undistort_self()
        cropped = self.temp_img_out[self.top_crop : self.bottom_crop, self.left_crop : self.right_crop]
        return cv2.resize(cropped, (self.width, self.height))


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


