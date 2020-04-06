import cv2
import sys
import os
from typing import List
from SJ5000_defisheyer import SJ5000_DeFisheyer

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
    #img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    defisheyer = SJ5000_DeFisheyer(img.shape[1], img.shape[0], 0)
    img_norm = defisheyer.process_frame(img)
    cv2.imwrite(file_out, img_norm)
    cv2.waitKey(-1)

if __name__ == "__main__":
    main(sys.argv)
