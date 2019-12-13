"""
This is not my work and everything is based on this great blog post:
https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

This package contains methods to perform object detections

Acknowledgments: pyimagesearch blog for his great work
"""
import time

import numpy as np
import cv2

from src.utils import utils


def object_detection_from_image(ssd_model, image, threshold_confidence):
    """
    Given a trained network and an image, perform object detection.
    :param ssd_model: trained network loaded through opencv
    :param image: opencv image element
    :param threshold_confidence: (float) minimum level of confidence to detect elements
    :return: opencv image with bounding boxes around detected objects, their class and confidence
    """
    # Keep track of image height and width then resize it to what is expected by the model we are working with
    h, w = image.shape[:2]
    size = ssd_model.get_size()
    img_resized = cv2.resize(image, (size, size))

    # Network is expecting a blob (https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)
    # We scale the image pixel values to a target range of 0 to 1 using a scale factor of 1/255=0.007843 (remember that
    # we multiply by this scale factor so it has to be 1/x)
    # Perform mean substraction with a mean value (here 255/2)
    # Normalization with values from ImageNet: (104, 117, 123)
    # Try also: [255/2, 255/2, 255/2]
    blob = cv2.dnn.blobFromImage(img_resized, scalefactor=0.007843, size=(size, size), mean=(104, 117, 123))

    # Perform a forward pass in the network
    print('Computing object detections...')
    ssd_model.get_net().setInput(blob)

    # Forward pass seems to be faster when output_names are given
    # Just a forward pass of the blob through the network to get the result (no backprop)
    last_layer = get_last_layer_name(ssd_model.get_net())
    start = time.time()
    outs = ssd_model.get_net().forward(last_layer)
    print('Found {} predictions'.format(outs[0].shape[2]))
    end = time.time()
    print("Forward pass took {:.5} seconds".format(end - start))

    # Loop over detections and handle those who are above a confidence threshold value
    # Detections within SSD is 4D where:
    #   * [2] is the number of detected elements
    #   * [3] is a tuple of 7 elements (0, class_id, confidence score, w, h, w, h for bounding box)
    for i in np.arange(0, outs[0].shape[2]):
        for detections in outs:
            confidence = detections[(0, 0, i, 2)]
            if confidence > threshold_confidence:
                idx = int(detections[(0, 0, i, 1)])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x_min, y_min, x_max, y_max = box.astype('int')

                # Print and put a label for each detected class
                label = '{}: {:.2f}%'.format(ssd_model.get_label(idx), confidence * 100)
                print('Found {} in picture!'.format(label))
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), ssd_model.get_color(idx), 2)

                # Draw label above label the rectangle when possible. If not, put it within the box
                y = y_min - 15 if y_min - 15 > 15 else y_min + 15
                utils.add_text_on_frame(image, label, (x_min, y), ssd_model.get_color(idx))

    return image


def get_last_layer_name(net):
    """
    Get the names of the output layers (i.e layers with unconnected outputs)
    :param net: (opencv network)
    :return: (string) name of the last layer of the network
    """
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
