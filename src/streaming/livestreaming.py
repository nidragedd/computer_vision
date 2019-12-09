"""
Based on this blog post: https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

Serve the output frames to our web browser via the Flask web framework. To display it we need to safely acquire locks
"""
import threading, cv2
from src.streaming import concurrent


class LiveStreamer(threading.Thread):

    def __init__(self, lock, use_gray=False):
        """
        Constructor for the streaming output thread
        :param lock: (theading lock) object to acquire and release to access frames
        :param use_gray: (boolean) if True we are reading the gray scale image from streaming queues
        """
        threading.Thread.__init__(self)
        self._lock = lock
        self._use_gray = use_gray

    def run(self):
        """
        This is a python generator that continuously transforms the output_frame into JPEG images that will be sent
        through Flask Response
        :return: a frame in a byte format
        """
        while True:
            self._lock.acquire()
            if self._use_gray:
                frame = concurrent.output_frame_gray
            else:
                frame = concurrent.output_frame
            self._lock.release()

            # Be careful, when there is no motion there is no gray frame to encode
            if frame is not None:
                flag, encoded_image = cv2.imencode('.jpg', frame)
                if not flag:
                    continue
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'
