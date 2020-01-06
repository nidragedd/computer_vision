"""
Main python script to call to try different models of SSD to perform object detection and then compare their
performance

@author: nidragedd
"""
import os
import cv2
import logging

from src.config import config
from src.config.pgconf import ProgramConfiguration
from src.detection.object_detection import ImageObjectDetector

logger = logging.getLogger("Object_Detection_Image")

if __name__ == '__main__':
    # Handle mandatory arguments
    args = config.parse_object_detection_image()
    config_file = vars(args)['config_file']
    log_config_file = vars(args)['log_file']

    # Configure the whole program (logging, external config files, singletons, ...)
    config.configure_logging(log_config_file)
    config.pgconf = ProgramConfiguration(config_file)

    img_dir = config.pgconf.get_images_dir()

    obj_detector = ImageObjectDetector()

    for model_name, ssd_model in config.pgconf.get_detection_models().items():
        logger.info("Working on object detection with SSD model '{}'".format(model_name))

        # Read a brand new version of the image, detect with SSD model and show output image for this SSD
        image = cv2.imread(os.path.join(img_dir, vars(args)["image"]))
        image = obj_detector.detect_with_model(ssd_model, image, vars(args)["confidence"])
        cv2.imshow("Detection with SSD model {}".format(model_name), image)

    cv2.waitKey(0)
