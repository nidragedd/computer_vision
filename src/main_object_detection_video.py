"""
Main python script to call to perform object detection on video streams

@author: nidragedd
"""
import os
import cv2
import logging

from src.config import config
from src.config.pgconf import ProgramConfiguration
from src.detection.object_detection import ImageObjectDetector
from src.utils import utils

logger = logging.getLogger("Object_Detection_Video")

if __name__ == '__main__':
    # Handle mandatory arguments
    args = config.parse_object_detection_video()
    config_file = vars(args)['config_file']
    log_config_file = vars(args)['log_file']

    # Configure the whole program (logging, external config files, singletons, ...)
    config.configure_logging(log_config_file)
    config.pgconf = ProgramConfiguration(config_file)

    vids_dir = config.pgconf.get_videos_dir()
    ssd_model = config.pgconf.get_detection_models()[vars(args)["model"]]

    # Init the video stream and try to count number of frames
    vs = cv2.VideoCapture(os.path.join(vids_dir, vars(args)["video"]))
    total_frames = utils.count_nb_frames(vs)

    logger.info("Found {} frames in the video".format(total_frames))

    obj_detector = ImageObjectDetector()

    # Continuously reading the frames and pass each one into model detector
    while True:
        # Pick one frame from video. If not grabbed then we have reached the end of the video stream
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # As for images, use detection model and confidence threshold
        image = obj_detector.detect_with_model(ssd_model, frame, vars(args)["confidence"])

        # Put image within a frame
        cv2.imshow("Video output", frame)

        # Stop if `q` key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release pointers
    cv2.destroyAllWindows()
    vs.release()
