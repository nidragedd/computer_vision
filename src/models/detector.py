"""
Kind of abstract class that is used to maintain common code for all detection process into a single place

@author: nidragedd
"""
import logging
from abc import abstractmethod

import cv2
import time


class AbstractDetector(object):
    def __init__(self, name):
        self._logger = logging.getLogger(name)
        self._model = None
        self._image = None
        self._threshold_confidence = None

    def detect_with_model(self, model, image, threshold_confidence):
        """
        Given a trained network and an image, perform detection task (object detection, pose estimation, ...)
        :param model: trained network loaded through opencv
        :param image: image element
        :param threshold_confidence: (float) minimum level of confidence to detect elements
        """
        # Keep references in memory so that it can be safely used in the subclass
        self._model = model
        self._image = image
        self._threshold_confidence = threshold_confidence

        # Used to perform mean substraction
        # Normalization with values from ImageNet: (123, 117, 104) if RGB
        mean = (123, 117, 104)
        # With Tensorflow pretrained models, samples do not use scaling, it does not work if we specify a scaling value
        scalefactor = 1

        # Handle BGR/RGB models
        if not model.is_RGB_model():
            mean = (104, 117, 123)
            # We scale the image pixel values to a target range of 0 to 1 using a scale factor of 1/255=0.007843
            # (remember that we multiply by this scale factor so it has to be 1/x)
            scalefactor = 1 / 255

        # Resize it to what is expected by the model we are working with
        size = model.get_size()
        img_resized = cv2.resize(image, (size, size))

        # Network expects a blob (https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)
        blob = cv2.dnn.blobFromImage(img_resized, scalefactor=scalefactor, size=(size, size), mean=mean,
                                     swapRB=model.is_RGB_model())

        # Perform a forward pass in the network
        self._logger.info('Model {} - Computing detections...'.format(self._model.get_name()))
        model.get_net().setInput(blob)

        # Forward pass seems to be faster when output_names are given
        last_layer = self._get_last_layer_name(model.get_net())

        # Just a forward pass of the blob through the network to get the result (no backprop)
        start = time.time()
        detections = model.get_net().forward(last_layer)
        end = time.time()
        self._logger.info("\tForward pass took {:.5} seconds".format(end - start))

        # Each subclass has to define what to do with detections, we return the result of this routine
        return self.handle_detections(detections)

    @abstractmethod
    def handle_detections(self, detections):
        """
        Each subclass has to define what to do with detected elements that comes out of the network forward pass
        :param detections: (object) detected elements
        """
        pass

    @staticmethod
    def _get_last_layer_name(net):
        """
        Get the names of the output layers (i.e layers with unconnected outputs)
        :param net: (opencv network)
        :return: (string) name of the last layer of the network
        """
        layers_names = net.getLayerNames()
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
