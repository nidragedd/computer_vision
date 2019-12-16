"""
Highly inspired by this blog post:
https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

This package contains methods to perform human pose detection

Acknowledgments: learnopencv blog for his great work
"""
import time
import logging
import numpy as np
import cv2

from src.utils import utils
from src.utils import constants as cst

logger = logging.getLogger("Pose_Estimation_Image")


def single_person_pose_detection_from_image(model, image, threshold_confidence, draw_skeleton=True):
    """
    Given a trained network and an image, perform human pose detection for a single person (WARNING: if multiple persons
    on the given image, the result will not be good, you have to use another approach).
    We check whether each keypoint is present in the image or not. We get the location of the keypoint by finding
    the max of the confidence map for that keypoint. We also use a threshold to reduce false detections.
    :param model: trained network loaded through opencv
    :param image: opencv image element
    :param threshold_confidence: (float) minimum level of confidence to detect elements
    :param draw_skeleton: (boolean) set to False to skip the skeleton drawing (default to True)
    :return: opencv image with keypoints, their class id
    """
    # Keep track of image height and width then resize it to what is expected by the model we are working with
    image_height, image_width = image.shape[:2]
    size = model.get_size()
    img_resized = cv2.resize(image, (size, size))

    # Network is expecting a blob (https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)
    # We scale the image pixel values to a target range of 0 to 1 using a scale factor of 1/255=0.007843 (remember that
    # we multiply by this scale factor so it has to be 1/x)
    # Perform mean substraction with a mean value (here 255/2)
    # Normalization with values from ImageNet: (104, 117, 123)
    # Try also: [255/2, 255/2, 255/2]
    blob = cv2.dnn.blobFromImage(img_resized, scalefactor=0.007843, size=(size, size), mean=(104, 117, 123))

    # Perform a forward pass in the network
    logger.info('Computing human pose estimation via keypoint detections...')
    model.get_net().setInput(blob)

    # Forward pass seems to be faster when output_names are given
    # Just a forward pass of the blob through the network to get the result (no backprop)
    last_layer = get_last_layer_name(model.get_net())
    start = time.time()
    outs = model.get_net().forward(last_layer)[0]
    end = time.time()
    logger.info('Found {} predictions'.format(outs.shape[1]))
    logger.info("Forward pass took {:.5} seconds".format(end - start))

    # Result of `forward` is a 4D matrix :
    #   * 1st dimension is image id
    #   * 2nd indicates the index of a keypoint. The model produces Confidence Maps and Part Affinity maps which are
    #   all concatenated:
    #       - For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity
    #       Maps.
    #       - Similarly, for MPI, it produces 44 points.
    #       ==> We use only the first few points which correspond to Keypoints.
    #   * 3rd & 4th dimension are respectively the height and width of the output map.

    h = outs.shape[2]
    w = outs.shape[3]
    keypoints = []

    # Iterate over the number of keypoints this model is able to detect
    for i in np.arange(0, model.get_nb_body_elements()):
        confidence_map = outs[0, i, :, :]

        # Find min, max location and value within the confidence map
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(confidence_map)

        # Scale the point to fit on the original image
        x = (image_width * max_loc[0]) / w
        y = (image_height * max_loc[1]) / h

        # Draw the keypoint and add coordinates to the list if confidence greater than the threshold
        if max_val > threshold_confidence:
            cv2.circle(image, (int(x), int(y)), 7, cst.FONT_COLOR_YELLOW, thickness=-1, lineType=cv2.FILLED)
            utils.add_text_on_frame(image, "{}".format(i), (int(x), int(y)), cst.FONT_COLOR_RED, font_thickness=2)

            keypoints.append((int(x), int(y)))
        else:
            keypoints.append(None)

    # Skeleton drawing by connecting dots that should be connected together
    if draw_skeleton:
        for pair in model.get_pose_pairs():
            body_element_a = pair[0]
            body_element_b = pair[1]
            if keypoints[body_element_a] and keypoints[body_element_b]:
                cv2.line(image, keypoints[body_element_a], keypoints[body_element_b], cst.FONT_COLOR_YELLOW, 3)

    # TODO: handle multi-person (https://github.com/spmallick/learnopencv/blob/master/OpenPose-Multi-Person/multi-person-openpose.py)

    return image


def get_last_layer_name(net):
    """
    Get the names of the output layers (i.e layers with unconnected outputs)
    :param net: (opencv network)
    :return: (string) name of the last layer of the network
    """
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
