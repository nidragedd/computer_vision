"""
Helper file used to load and configure external configuration files

@author: nidragedd
"""
import os
import json
import logging.config
import argparse

# Single ref to configuration over the whole program
pgconf = None


def configure_logging(log_config_file):
    """
    Setup logging configuration (il file cannot be loaded or read, a fallback basic configuration is used instead)
    :param log_config_file: (string) path to external logging configuration file
    """
    if os.path.exists(log_config_file):
        with open(log_config_file, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)


def parse_object_detection_image():
    """
    Parse main program arguments to ensure everything is correctly launched
    :return: argparse object configured with mandatory and optional arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config-file', required=True, help="External configuration file (JSON format)")
    ap.add_argument('-l', '--log-file', required=True, help="External logging configuration file (JSON format)")
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-t", "--confidence", type=float, default=0.1, help="minimum probability to filter weak detections")

    return ap.parse_args()


def parse_object_detection_video():
    """
    Parse main program arguments to ensure everything is correctly launched
    :return: argparse object configured with mandatory and optional arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config-file', required=True, help="External configuration file (JSON format)")
    ap.add_argument('-l', '--log-file', required=True, help="External logging configuration file (JSON format)")
    ap.add_argument("-v", "--video", required=True, help="path to input video file")
    ap.add_argument("-m", "--model", required=True, help="key name for the model to use")
    ap.add_argument("-t", "--confidence", type=float, default=0.1, help="minimum probability to filter weak detections")

    return ap.parse_args()


def parse_applaunch_args():
    """
    Parse main program arguments when launching the webapp
    :return: argparse object configured with mandatory and optional arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config-file', required=True, help="External configuration file (JSON format)")
    ap.add_argument('-l', '--log-file', required=True, help="External logging configuration file (JSON format)")
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-p", "--port", type=int, required=True, help="server port number")

    return ap.parse_args()