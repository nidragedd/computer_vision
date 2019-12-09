"""
This package is a model zoo for all models that have been downloaded from different places

@author: nidragedd
"""
import os
import cv2
import numpy as np
from src.utils import constants as cst


class SSDModel(object):
    """
    This class is a wrapper for our different Single Shot Detector available models
    """

    def __init__(self, proto_file, model_file, size, classes_file):
        """
        Constructor
        :param proto_file: (string) absolute path to the 'deploy' proto.txt file
        :param model_file: (string) absolute path to the model file (might be a caffe model, tensorflow, whatever)
        :param size: (int) the size for input image for this SSD
        :param classes_file: (string) absolute path to the file that contains all classes
        """
        if model_file.split('.')[-1:][0] == 'caffemodel':
            self._net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
        self._size = size
        # TODO: fail if classes_file does not exist
        with open(classes_file, 'rt') as f:
            self._classes = f.read().rstrip('\n').split('\n')
        self._colors = np.random.uniform(0, 255, size=(len(self._classes), 3))

    def get_net(self):
        """
        :return: (object) the opencv loaded network
        """
        return self._net

    def get_label(self, idx):
        """
        :param idx: (int) the indice of the detected object
        :return: the label for the detected object
        """
        return self._classes[idx]

    def get_color(self, idx):
        """
        :param idx: (int) the indice of the detected object
        :return: a random color generated at the construction step
        """
        return self._colors[idx]

    def get_size(self):
        """
        :return: (int) size the input image should be resized to in order to use this model
        """
        return self._size


SSD_MODELS = {'MobileNet_300_PASCALVOC12': SSDModel(os.path.join(cst.MOBILENET_300_PASCALVOC12, 'deploy.prototxt'),
                                                    os.path.join(cst.MOBILENET_300_PASCALVOC12,
                                                                 'MobileNetSSD_deploy.caffemodel'), 300,
                                                    cst.PASCALVOC2012_CLASSES),
              'VGG_300_COCO': SSDModel(os.path.join(cst.VGG_300_COCO, 'deploy.prototxt'),
                                       os.path.join(cst.VGG_300_COCO, 'VGG_coco_SSD_300x300_iter_400000.caffemodel'),
                                       300, cst.COCO_CLASSES),
              'VGG_300_ILSVRC16': SSDModel(os.path.join(cst.VGG_300_ILSVRC16, 'deploy.prototxt'),
                                           os.path.join(cst.VGG_300_ILSVRC16,
                                                        'VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel'), 300,
                                           cst.ILSVRC16_CLASSES)}