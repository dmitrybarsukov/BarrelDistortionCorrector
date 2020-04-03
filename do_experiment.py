import cv2
import streamlit as st
import numpy as np
from frame_processor import process_frame

def add_border(img: np.ndarray, part: float) -> np.ndarray:
    height, width, depth = img.shape
    newheight, newwidth = int(height * (1 + part * 2)), int(width * (1 + part * 2))
    dh, dw = (newheight - height) // 2, (newwidth - width) // 2
    newimg = np.zeros((newheight, newwidth, depth), dtype=img.dtype)
    newimg[dh : dh + height, dw : dw + width] = img
    return newimg

def rm_border(img: np.ndarray, shape: (int, int)) -> np.ndarray:
    shape = int(shape[0] * 1.1), int(shape[1] * 1.1)
    height, width, depth = img.shape
    dh, dw = (height - shape[0]) // 2, (width - shape[1]) // 2
    #cv2.rectangle(img, (dw, dh), (dw + shape[1], dh + shape[0]), (0, 255, 0), 5)
    return cv2.resize(img, (shape[1], shape[0]))
    #return #img[dh : dh + shape[0], dw : dw + shape[1]].copy()

img = cv2.imread("samples\\2020_0401_212026_005.JPG")
img2 = add_border(img, 0.08) # 0.08 for 4/3, 0.05 for 16:9
img2 = process_frame(img2, -1.3, 1.25)
img2 = rm_border(img2, img.shape[0:2])
st.image(img, use_column_width=True, channels='BGR')
st.image(img2, use_column_width=True, channels='BGR')
