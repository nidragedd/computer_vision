"""
This is not only my work and the detection code is mostly based on this great blog post:
https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

I have factorized the code which is common with other detections techniques (Pose Estimation for example) by using
classes

This script contains methods to perform object detections

Acknowledgments: pyimagesearch blog for his great work

@author: nidragedd
"""
import numpy as np
import cv2

from src.models.detector import AbstractDetector
from src.utils import utils


class ImageObjectDetector(AbstractDetector):
    """
    This detector uses models to detect object within images. It displays bouding boxes and class around detected
    elements
    """
    def __init__(self):
        super().__init__("Object_Detection_Image")

    def handle_detections(self, outs):
        self._logger.info('\tFound {} predictions'.format(outs[0].shape[2]))

        # Keep track of image height and width
        h, w = self._image.shape[:2]

        # Loop over detections and handle those who are above a confidence threshold value
        # Detections within SSD is 4D where:
        #   * [2] is the number of detected elements
        #   * [3] is a tuple of 7 elements (0, class_id, confidence score, w, h, w, h for bounding box)
        for i in np.arange(0, outs[0].shape[2]):
            for detections in outs:
                confidence = detections[(0, 0, i, 2)]
                if confidence > self._threshold_confidence:
                    idx = int(detections[(0, 0, i, 1)])

                    # Find box coordinates and draw it on picture
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x_min, y_min, x_max, y_max = box.astype('int')
                    cv2.rectangle(self._image, (x_min, y_min), (x_max, y_max), self._model.get_color(idx), 2)

                    # Log and add a label for each detected class
                    label = '{}: {:.2f}%'.format(self._model.get_label(idx), confidence * 100)
                    self._logger.info('\t\tFound {} in picture!'.format(label))

                    # Draw class label above the box when possible. If not, put it within it
                    y = y_min - 15 if y_min - 15 > 15 else y_min + 15
                    utils.add_text_on_frame(self._image, label, (x_min, y), self._model.get_color(idx))

        return self._image.copy()
