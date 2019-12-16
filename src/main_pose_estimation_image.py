"""
Main python script to call to try different models of SSD to perform object detection and then compare their
performance

@author: nidragedd
"""
import os
import cv2

from src.config import config
from src.config.pgconf import ProgramConfiguration
from src.pose import pose_detection

if __name__ == '__main__':
    # Handle mandatory arguments
    args = config.parse_object_detection_image()
    config_file = vars(args)['config_file']
    log_config_file = vars(args)['log_file']

    # Configure the whole program (logging, external config files, singletons, ...)
    config.configure_logging(log_config_file)
    config.pgconf = ProgramConfiguration(config_file)

    img_dir = config.pgconf.get_images_dir()

    for model_name, pose_model in config.pgconf.get_pose_estimation_models().items():
        # Read a brand new version of the image, detect with Pose Estimation model and show output image
        image = cv2.imread(os.path.join(img_dir, vars(args)["image"]))
        image = pose_detection.single_person_pose_detection_from_image(pose_model, image, vars(args)["confidence"])
        cv2.imshow("Detection with Pose model {}".format(model_name), image)

    cv2.waitKey(0)
