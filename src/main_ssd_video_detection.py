"""
Main python script to call to perform object detection on video streams

@author: nidragedd
"""
import os
import argparse

import cv2

from src.detection.models import model_zoo
from src.detection.object import object_detection
from src.utils import utils

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--video", required=True, help="path to input video")
    ap.add_argument("-m", "--model", required=True, help="key name for the model to use")
    ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    vids_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'data', 'videos'))

    ssd_model = model_zoo.SSD_MODELS[args["model"]]

    # Init the video stream and try to count number of frames
    vs = cv2.VideoCapture(os.path.join(vids_dir, args["video"]))
    total_frames = utils.count_nb_frames(vs)
    print("Found {} frames in the video".format(total_frames))

    # Continuously reading the frames and pass each one into model detector
    while True:
        # Pick one frame from video. If not grabbed then we have reached the end of the video stream
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # As for images, use detection model and confidence threshold
        image = object_detection.object_detection_from_image(ssd_model, frame, args["confidence"])

        # Put image within a frame
        cv2.imshow("Video output", frame)

        # Stop if `q` key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release pointers
    cv2.destroyAllWindows()
    vs.release()
