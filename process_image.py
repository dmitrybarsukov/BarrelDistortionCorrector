import cv2
import sys
import os
from typing import List
from frame_processor import process_frame


def main(args: List[str]):
    for f in args[1:]:
        file_in = os.path.abspath(f)
        file_out = make_processed_name(file_in)
        process_image(file_in, file_out)


def make_processed_name(file_in: str) -> str:
    file_name, extension = os.path.splitext(file_in)
    return f"{file_name}-processed{extension}"


def process_image(file_in: str, file_out: str):
    img = cv2.imread(file_in)
    size = (img.shape[1] // 4, img.shape[0] // 4)
    img = cv2.resize(img, size)
    img_norm = process_frame(img, (-0.00006, 0.0, 0.0, 0.0), (10.0, 10.0))
    cv2.imshow("Raw", img)
    cv2.imshow("Norm", img_norm)
    cv2.waitKey(-1)


if __name__ == "__main__":
    main(sys.argv)
