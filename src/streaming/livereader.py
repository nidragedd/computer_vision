import threading, cv2, time
from src.detection.motion.livemotiondetector import SingleMotionDetector
from src.streaming import concurrent
from src.utils import utils
from src.utils import constants as cst


class LiveReader(threading.Thread):
    """
    A thread that is launched to continuously read frames from input
    """

    def __init__(self, frame_count, videostream, lock, gray_lock):
        """
        Constructor
        :param frame_count: (int) number of frames to read before starting motion detection
        :param videostream: (object) reference to the videostream object to start
        """
        threading.Thread.__init__(self)
        self._frame_count = frame_count
        self._vs = videostream.start()
        # Allow some warmup
        time.sleep(2.0)
        self._lock = lock
        self._gray_lock = gray_lock

    def run(self):
        """
        This replaces the output frame sent to server (for that we need to safely acquire a lock element)
        """
        single_motion_detector = SingleMotionDetector(accum_weight=0.1)
        total_nb_frames_read = 0
        while True:
            frame, gray = utils.get_converted_frame(self._vs)
            thresh = None
            utils.add_datetime_on_frame(frame)

            if total_nb_frames_read > self._frame_count:
                # Use detector to detect motion in the gray image
                motion = single_motion_detector.detect(gray)

                # If motion if found then we display a big red bounding box surrounding the biggest motion area
                # For smaller but still higher than a threshold value we display green bouding boxes
                if motion is not None:
                    thresh, (minX, minY, maxX, maxY), img_contours = motion
                    cv2.rectangle(frame, (minX, minY), (maxX, maxY), cst.FONT_COLOR_RED, 2)
                    for c in img_contours:
                        if cv2.contourArea(c) < cst.LIVE_MOTION_DETECTION_MIN_AREA:
                            continue
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), cst.FONT_COLOR_GREEN, 2)

            single_motion_detector.update(gray)
            total_nb_frames_read += 1
            self._lock.acquire()
            concurrent.output_frame = frame.copy()
            self._lock.release()

            # If no motion then there is no threshold image to compute
            if thresh is not None:
                self._gray_lock.acquire()
                concurrent.output_frame_gray = thresh.copy()
                self._gray_lock.release()
