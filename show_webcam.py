import cv2
import sys
import numpy as np
from typing import List
from SJ5000_defisheyer import SJ5000_DeFisheyer

def main(args: List[str]):
    cap = cv2.VideoCapture(1)
    _, img = cap.read()
    defisheyer = SJ5000_DeFisheyer(img.shape[1], img.shape[0], 0)
    while True:
        _, img = cap.read()
        if img is None:
            break

        norm_img = defisheyer.process_frame(img)

        cv2.imshow("Raw", img)
        cv2.imshow("Norm", norm_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
