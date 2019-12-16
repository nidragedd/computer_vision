"""
Class used to handle program configuration

@author: nidragedd
"""
import logging
import os
import json
import src.utils.constants as cst
from src.models.objects import SSDModel, PoseEstimationModel


class ProgramConfiguration(object):
    """
    Class used to handle and maintain all parameters of this program (timeouts, some other values...)
    """
    _logger = logging.getLogger()

    def __init__(self, config_file_path):
        """
        Constructor - Loads the given external JSON configuration file. Raises an error if not able to do it.
        :param config_file_path: (string) full path to the JSON configuration file
        """
        if os.path.exists(config_file_path):
            self._config_directory = os.path.dirname(config_file_path)
            with open(config_file_path, 'rt') as f:
                self._config = json.load(f)
                self._logger.info(
                    "External JSON configuration file ('{}') successfully loaded".format(config_file_path))
        else:
            raise Exception("Could not load external JSON configuration file '{}'".format(config_file_path))
        self._ssd_models = None
        self._pose_models = None

    @staticmethod
    def _get_path_to_data_static_subdirectory(dir_name):
        """
        :param dir_name: (string) name of the directory within static/data
        :return: (string) absolute path to local folder that contains static data elements
        """
        return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'static', 'data', dir_name))

    def _get_path_to_folder(self, folder_key):
        """
        Get full path to a folder given its key name in the external JSON configuration file
        :param folder_key: key name of the folder as written in JSON configuration file
        :return: (string) absolute path to folder
        """
        # If path to raw folder is relative to JSON configuration file
        if folder_key.startswith('../'):
            folder_key = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, folder_key)
        elif folder_key.startswith('./'):
            # There is a trick here as the abspath is related to this .py file and not the .json file but we want to act
            # like if it were the case for the user (so if he puts ./ it should be related to config.json file
            folder_key = os.path.join(self._config_directory, folder_key)
        if os.path.exists(folder_key):
            return os.path.abspath(folder_key)
        else:
            raise Exception("Directory '{}' does not exist".format(folder_key))

    def _get_models_dir_path(self, model_domain):
        """
        :param model_domain: (string) an identifier of the domain this model is related to ('classification', 'pose
        estimation', 'image segmentation', etc)
        :return: (string) absolute path to the local folder where serialized pre-trained model are stored
        """
        return self._get_path_to_folder(self._config[model_domain]["models_dir_name"])

    def _get_config_dir_path(self, model_domain):
        """
        :param model_domain: (string) an identifier of the domain this model is related to ('classification', 'pose
        estimation', 'image segmentation', etc)
        :return: (string) absolute path to the local folder where classes file for the model domain are stored
        """
        return self._get_path_to_folder(self._config[model_domain]["config_dir_name"])

    def _get_detection_models_dir_path(self):
        """
        :return: (string) absolute path to the local folder where serialized pre-trained object detection models are
        stored
        """
        return self._get_models_dir_path(cst.DETECT_CONFIG_JSON_KEY)

    def _get_pose_estimation_models_dir_path(self):
        """
        :return: (string) absolute path to the local folder where serialized pre-trained pose estimation models are
        stored
        """
        return self._get_models_dir_path("pose_estimation")

    def _get_detection_classes_config_dir_path(self):
        """
        :return: (string) absolute path to the local folder where classes files for object detection models are stored
        """
        return self._get_config_dir_path(cst.DETECT_CONFIG_JSON_KEY)

    def _get_pose_estimation_classes_config_dir_path(self):
        """
        :return: (string) absolute path to the local folder where classes files for pose estimation models are stored
        """
        return self._get_config_dir_path("pose_estimation")

    def _load_detection_models(self):
        """
        Load serialized pre-trained object detection models and store them into a dict object
        """
        self._ssd_models = {}
        for models in self._config[cst.DETECT_CONFIG_JSON_KEY]['models']:
            model_dir_path = os.path.join(self._get_detection_models_dir_path(),
                                          models["trained_on_dataset"],
                                          models["folder"])
            classes_file_name = self._config[cst.DETECT_CONFIG_JSON_KEY]["classes"][models["trained_on_dataset"]]

            if models["type"] == cst.SSD_MODEL_TYPE:
                model = SSDModel(models["name"],
                                 os.path.join(model_dir_path, models["proto_file"]),
                                 os.path.join(model_dir_path, models["weights_file"]),
                                 models["image_input_size"],
                                 os.path.join(self._get_detection_classes_config_dir_path(), classes_file_name))
                self._ssd_models[models["name"]] = model

    def _load_pose_estimation_models(self):
        """
        Load serialized pre-trained pose estimation models and store them into a dict object
        """
        self._pose_models = {}
        for models in self._config[cst.POSE_CONFIG_JSON_KEY]['models']:
            model_dir_path = os.path.join(self._get_pose_estimation_models_dir_path(),
                                          models["trained_on_dataset"],
                                          models["folder"])
            classes_file_name = self._config[cst.POSE_CONFIG_JSON_KEY]["classes"][models["trained_on_dataset"]]
            pairs = self._config[cst.POSE_CONFIG_JSON_KEY]["pairs"][models["trained_on_dataset"]]

            model = PoseEstimationModel(models["name"],
                                        os.path.join(model_dir_path, models["proto_file"]),
                                        os.path.join(model_dir_path, models["weights_file"]),
                                        models["image_input_size"],
                                        os.path.join(self._get_pose_estimation_classes_config_dir_path(),
                                                     classes_file_name), pairs)
            self._pose_models[models["name"]] = model

    def get_detection_models(self):
        """
        :return: (dict) all object detection models that were loaded
        """
        # Load only once (lazy initialization)
        if not self._ssd_models:
            self._load_detection_models()
        return self._ssd_models

    def get_pose_estimation_models(self):
        """
        :return: (dict) all pose_estimation models that were loaded
        """
        # Load only once (lazy initialization)
        if not self._pose_models:
            self._load_pose_estimation_models()
        return self._pose_models

    def get_images_dir(self):
        """
        :return: (string) full path to images folder within static data folder
        """
        return self._get_path_to_data_static_subdirectory('images')

    def get_videos_dir(self):
        """
        :return: (string) full path to videos folder within static data folder
        """
        return self._get_path_to_data_static_subdirectory('videos')
