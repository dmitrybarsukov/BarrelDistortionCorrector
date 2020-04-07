import cv2
import sys
import os
from typing import List
from SJ5000_defisheyer import SJ5000_DeFisheyer

debug = True

def main(args: List[str]):
    for f in args[1:]:
        file_in = os.path.abspath(f)
        file_out = make_processed_name(file_in)
        process_video(file_in, file_out)

def make_processed_name(file_in: str) -> str:
    file_name, extension = os.path.splitext(file_in)
    return f"{file_name}-processed.avi"

def process_video(file_in: str, file_out: str):
    video = cv2.VideoCapture(file_in)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = video.get(cv2.CAP_PROP_FPS)
    defisheyer = SJ5000_DeFisheyer(frame_width, frame_height, rotation=0)
    video_new_size = defisheyer.get_new_frame_size()
    out = cv2.VideoWriter(file_out, cv2.VideoWriter_fourcc('H','2','6','4'), frame_fps, video_new_size)
    while True:
        _, frame = video.read()
        if frame is None:
            break
        norm_frame = defisheyer.process_frame(frame)
        out.write(norm_frame)
        if debug:
            #cv2.imshow("original", cv2.resize(frame, (960, 540)))
            #cv2.imshow("normalized", cv2.resize(norm_frame, (960, 540)))
            cv2.imshow("normalized", cv2.resize(norm_frame, (norm_frame.shape[1] // 2, norm_frame.shape[0] // 2)))
            cv2.waitKey(1)

    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
