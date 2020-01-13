"""
Highly inspired by those 2 blog posts:
https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
https://www.learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose/

This package contains methods to perform human pose detection with an OpenPose model

Acknowledgments: learnopencv blog for their great work
"""
import numpy as np
import cv2

from src.models.detector import AbstractDetector
from src.utils import utils, geometry
from src.utils import constants as cst


class HumanPoseDetector(AbstractDetector):
    """
    This detector uses models to detect human pose within images. It displays keypoints and class and eventually a
    skeleton
    """
    def __init__(self, draw_skeleton=True):
        """
        Constructor
        :param draw_skeleton: (boolean) set to False to skip the skeleton drawing (default to True)
        """
        super().__init__("OpenPose_Estimation_Image")
        self._draw_skeleton = draw_skeleton

    def handle_detections(self, detections):
        outs = detections[0]
        self._logger.info('\tFound {} predictions'.format(outs.shape[1]))

        # Result of `forward` is a 4D matrix :
        #   * 1st dimension is image id
        #   * 2nd indicates the index of a keypoint. The model produces Confidence Maps and Part Affinity maps which
        #   are all concatenated:
        #       - For COCO model it consists of 57 parts: 18 keypoint confidence Maps + 1 background
        #       + 19*2 Part Affinity Maps.
        #       - Similarly, for MPII, it produces 44 points.
        #       ==> We use only the first few points which correspond to Keypoints.
        #   * 3rd & 4th dimension are respectively the height and width of the output map.
        image_height, image_width = self._image.shape[:2]
        h = outs.shape[2]
        w = outs.shape[3]
        keypoints = []

        # Iterate over the number of keypoints this model is able to detect
        for i in np.arange(0, self._model.get_nb_body_elements()):
            confidence_map = outs[0, i, :, :]

            # Find min, max location and value within the confidence map
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(confidence_map)

            # Scale the point to fit on the original image
            x = (image_width * max_loc[0]) / w
            y = (image_height * max_loc[1]) / h

            # Draw the keypoint and add coordinates to the list if confidence greater than the threshold
            if max_val > self._threshold_confidence:
                cv2.circle(self._image, (int(x), int(y)), 7, cst.FONT_COLOR_YELLOW, thickness=-1, lineType=cv2.FILLED)
                utils.add_text_on_frame(self._image, "{}".format(i), (int(x), int(y)), cst.FONT_COLOR_RED,
                                        font_thickness=2)

                keypoints.append((int(x), int(y)))
            else:
                keypoints.append(None)

        self._handle_skeleton(keypoints)

        return self._image.copy()

    def _handle_skeleton(self, keypoints):
        """
        Draw skeleton by connecting the dots that should be connected together
        :param keypoints: (array) positions (x, y) for each detected keypoint, None if not detected
        """
        if self._draw_skeleton:
            for pair in self._model.get_pose_pairs():
                body_element_a = pair[0]
                body_element_b = pair[1]
                if keypoints[body_element_a] and keypoints[body_element_b]:
                    cv2.line(self._image, keypoints[body_element_a], keypoints[body_element_b],
                             cst.FONT_COLOR_YELLOW, 3)

            self._handle_angles_checking(keypoints)

    def _handle_angles_checking(self, keypoints):
        """
        Compute angle between some specific body parts
        :param keypoints: (array) positions (x, y) for each detected keypoint, None if not detected
        """
        # Do it only if possible (eg. keypoints have been found with enough confidence
        if keypoints[0] and keypoints[1] and keypoints[14]:
            head_neck_chest = [keypoints[0], keypoints[1], keypoints[14]]
            angle = geometry.get_angle_degree(head_neck_chest)

            self._logger.info("\tAngle between head, neck and chest is {} degrees".format(np.degrees(angle)))
