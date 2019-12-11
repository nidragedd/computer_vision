"""
This package contains utility and helper functions to manipulate images, streams, etc

@author: nidragedd
"""
import imutils
from imutils import paths
from datetime import datetime
import cv2
import os
from src.utils import constants as cst


def get_converted_frame(videostream):
    """
    Read a new frame from the video stream, resize and convert into gray scale.
    Apply a small gaussian blur to remove noise
    :param videostream: imutils videostream object instance
    (https://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/)
    :return: (tuple) 1st element is the read frame, 2nd is the converted grayscale image
    """
    frame = videostream.read()
    frame = imutils.resize(frame, width=cst.FRAME_RESIZED_VALUE)
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscale_frame = cv2.GaussianBlur(grayscale_frame, (7, 7), 0)

    return frame, grayscale_frame


def add_datetime_on_frame(frame):
    """
    Add a timestamp text on the frame
    :param frame: the frame on which the text will be added
    """
    timestamp = datetime.now().strftime('%A %d %B %Y %I:%M:%S%p')
    corner_origin = (10, frame.shape[0] - 10)
    cv2.putText(frame, timestamp, corner_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.35, cst.FONT_COLOR_RED, 1)


def get_list_img_files(dir_name):
    """
    Get a list of images available in a given directory
    :param dir_name: (string) directory to explore
    :return: (array) list of file names (only file name, not absolute path). If you want absolute path then use the
    imutils.paths.list_images function that returns a generator
    """
    file_list = []
    for _, _, files in os.walk(dir_name):
        for a_file in files:
            ext = a_file[a_file.rfind('.'):].lower()
            if ext.endswith(paths.image_types):
                file_list.append(a_file)

        return file_list


def count_nb_frames(videostream):
    """
    Try to find the total number of frames given a video stream. -1 is returned if not able to find it
    :param videostream: (imutils video stream element)
    :return: nb of frames in video, -1 if not able to catch it
    """
    total = 0
    try:
        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(videostream.get(prop))
    except:
        total = -1
    return total
