import cv2
import sys
import numpy as np
from typing import List
from frame_processor import process_frame

def main(args: List[str]):
    cap = cv2.VideoCapture(1)
    while True:
        _, img = cap.read()
        if img is None:
            break

        norm_img = process_frame(img, (0.5, 0.5), (-0.00015, 0.0, 0.0, 0.0), (10.0, 10.0))

        cv2.imshow("Raw", img)
        cv2.imshow("Norm", norm_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
