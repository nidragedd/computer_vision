"""
This thread class is highly similar to the one that is used for live motion detection. It uses an object detection model
to detect object within a videostream.

@author: nidragedd
"""
import threading
import time

from src.detection.object import object_detection
from src.utils import utils, concurrent
from src.utils import constants as cst


class LiveObjectDetector(threading.Thread):
    """
    A thread that is launched to continuously read frames from input
    """

    def __init__(self, videostream, model, lock):
        """
        Constructor
        :param videostream: (object) reference to the videostream object to start
        :param model: (object) reference to the model to use for object detection
        :param lock: (object) reference to the thread lock to acquire
        """
        threading.Thread.__init__(self)
        self._vs = videostream.start()
        time.sleep(cst.VIDEOSTREAM_WARMUP)
        self._model = model
        self._lock = lock
        self._status = cst.RUNNING_STREAM_STATUS

    def run(self):
        """
        This replaces the output frame sent to server (for that we need to safely acquire a lock element)
        """
        while True and self._status == cst.RUNNING_STREAM_STATUS:
            frame, _ = utils.get_converted_frame(self._vs)
            image = object_detection.object_detection_from_image(self._model, frame, cst.CONFIDENCE_THRESHOLD)

            self._lock.acquire()
            concurrent.object_detection_output_frame = image.copy()
            self._lock.release()

    def quit(self):
        """
        Expose a way to stop this thread once it has been started
        """
        print("Quit called in Live Object Detector")
        self._status = "quit"
        self._vs.stop()
