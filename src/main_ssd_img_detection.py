"""
Main python script to call to try different models of SSD to perform object detection and then compare their
performance

@author: nidragedd
"""
import os
import argparse
import cv2

from src.detection.models import model_zoo
from src.detection.object import object_detection

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'data', 'images'))

    for model_name, ssd_model in model_zoo.SSD_MODELS.items():
        # Read a brand new version of the image, detect with SSD model and show output image for this SSD
        image = cv2.imread(os.path.join(img_dir, args["image"]))
        image = object_detection.object_detection_from_image(ssd_model, image, args["confidence"])
        cv2.imshow("Detection with SSD model {}".format(model_name), image)

    cv2.waitKey(0)
