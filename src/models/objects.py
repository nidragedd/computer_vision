"""
This package holds different type of models that have been downloaded from different places

@author: nidragedd
"""
import cv2
import numpy as np


class CVModel(object):
    def __init__(self, name, proto_file, model_file, size, classes_file):
        """
        Constructor
        :param name: (string) name identifier of the model
        :param proto_file: (string) absolute path to the 'deploy' proto.txt file
        :param model_file: (string) absolute path to the model file (might be a caffe model, tensorflow, whatever)
        :param size: (int) the size for input image for this SSD
        :param classes_file: (string) absolute path to the file that contains all classes
        """
        self._name = name
        self._is_RGB_model = True
        model_type = model_file.split('.')[-1:][0]

        if model_type == 'caffemodel':
            self._net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
            self._is_RGB_model = False
        elif model_type == "pb":
            self._net = cv2.dnn.readNetFromTensorflow(model_file, proto_file)
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

    def get_name(self):
        """
        :return: (string) identifier of the model
        """
        return self._name

    # noinspection PyPep8Naming
    def is_RGB_model(self):
        """
        :return: True if the model has been trained with images in RGB format. If True, it might mean that we need to
        convert from BGR to RGB before feeding into the network (for example with Tensorflow models)
        """
        return self._is_RGB_model


class SSDModel(CVModel):
    """
    This class is a wrapper for our different Single Shot Detector available models
    """
    def get_label(self, idx):
        """
        :param idx: (int) the indice of the detected object
        :return: the label for the detected object
        """
        return self._classes[idx]


class PoseEstimationModel(CVModel):
    """
    This class is a wrapper for our different Human Pose Estimation available models
    """

    def __init__(self, name, proto_file, model_file, size, classes_file, pose_pairs):
        """
        Constructor
        :param name: (string) name identifier of the model
        :param proto_file: (string) absolute path to the 'deploy' proto.txt file
        :param model_file: (string) absolute path to the model file (might be a caffe model, tensorflow, whatever)
        :param size: (int) the size for input image for this model
        :param classes_file: (string) absolute path to the file that contains all classes
        :param pose_pairs: (array) array of arrays corresponding to pose pairs if we want to draw skeleton
        """
        super().__init__(name, proto_file, model_file, size, classes_file)
        self._pose_pairs = pose_pairs

    def get_nb_body_elements(self):
        """
        :return: (int) the number of body elements this model is able to detect
        """
        # We remove 1 because one class is the background and we should not take care of it
        return len(self._classes) - 1

    def get_pose_pairs(self):
        """
        :return: (array) array of arrays corresponding to pose pais if we want to draw skeleton
        """
        return self._pose_pairs
