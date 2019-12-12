import base64

from flask import render_template, request
from flask import Response
from imutils.video import VideoStream
import os
import cv2

from src import app
from src.detection.models import model_zoo
from src.detection.object import object_detection
from src.utils import utils, concurrent
from src.utils import constants as cst
from src.motion.livestreaming import LiveStreamer
from src.motion.livemotiondetector import LiveMotionDetector
from src.detection.liveobjectdetector import LiveObjectDetector

live_motion_detector_t = None
live_object_detector_t = None


@app.route('/')
def index():
    """
    Display the homepage. HTML file has to be placed within a 'templates' folder near by this module file as per
    Flask conventions
    """
    stop_all_streams()
    return render_template('master.html')


@app.route('/img_carousel', methods=['GET', 'POST'])
def image_carousel():
    stop_all_streams()
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'static', 'data', 'images'))
    img_list = utils.get_list_img_files(img_dir)
    car_item = request.args.get('car_item') if request.args.get('car_item') else 0
    return render_template('image_carousel.html', firstid=(img_list[0]), img_list=(img_list[1:]), car_item=car_item)


@app.route('/do_img_detection', methods=['POST'])
def launch_obj_detection():
    # TODO: handle confidence threshold as a request arg (with something like a slider or whatever
    confidence = cst.CONFIDENCE_THRESHOLD

    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'static', 'data', 'images'))
    img_id = request.form['picked_img_id']
    img_url = request.form['picked_img_url']

    model_names = []
    img_generated = []

    for model_name, ssd_model in model_zoo.SSD_MODELS.items():
        model_names.append(model_name)

        # Read a brand new version of the image, detect with SSD model and show output image for this SSD
        image = cv2.imread(os.path.join(img_dir, img_url.split('/')[-1:][0]))
        image = object_detection.object_detection_from_image(ssd_model, image, confidence)

        # Transform as byte array and then as Base64 image
        flag, encoded_image = cv2.imencode('.jpg', image)
        b64_img = base64.b64encode(bytearray(encoded_image)).decode('utf-8')

        img_generated.append(b64_img)

    # Zip model names and the corresponding generated image
    images_data = zip(model_names, img_generated)

    return render_template('image_detection.html', img_id=img_id, img_url=img_url, images_data=images_data)


@app.route('/about')
def about():
    stop_all_streams()
    return render_template('about.html')


@app.route('/object_detection_live')
def livestream_object_detection():
    """
    Display live object detection page
    """
    # TODO: here the object detection model should be something given as request argument or something like that
    ssd_model = model_zoo.SSD_MODELS["MobileNet_300_PASCALVOC12"]
    stop_live_motion_stream()
    global live_object_detector_t
    if not live_object_detector_t:
        live_object_detector_t = LiveObjectDetector(VideoStream(src=0), ssd_model, concurrent.object_detection_lock)
        live_object_detector_t.start()
    return render_template('livestream_object_detection.html')


@app.route('/motion_detection_live')
def livestream_motion_detection():
    """
    Display live motion detection page.
    """
    stop_live_object_stream()
    global live_motion_detector_t
    if not live_motion_detector_t:
        live_motion_detector_t = LiveMotionDetector(32, VideoStream(src=0), concurrent.motion_detection_lock,
                                                    concurrent.motion_detection_lock_gray)
        live_motion_detector_t.start()
    return render_template('livestream_motion_detection.html')


@app.route('/motion_detection_video_feed')
def motion_detection_video_feed():
    """
    Use a generator to return the response generated along with the specific media type (mime type)
    :return: a Flask Response element wrapping a frame containing image and bounding box around detected motion
    """
    t = LiveStreamer(concurrent.motion_detection_lock, cst.STREAM_MOTION)
    return Response((t.run()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/motion_detection_video_feed_delta')
def motion_detection_video_feed_delta():
    """
    Use a generator to return the response generated along with the specific media type (mime type)
    :return: a Flask Response element wrapping a frame containing image and bounding box around detected motion
    """
    t = LiveStreamer(concurrent.motion_detection_lock_gray, cst.STREAM_MOTION, use_gray=True)
    return Response((t.run()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/object_detection_video_feed')
def object_detection_video_feed():
    """
    Use a generator to return the response generated along with the specific media type (mime type)
    :return: a Flask Response element wrapping a frame containing image and bounding box around detected objects
    """
    t = LiveStreamer(concurrent.object_detection_lock, cst.STREAM_OBJECT)
    return Response((t.run()), mimetype='multipart/x-mixed-replace; boundary=frame')


def stop_all_streams():
    """
    Call this method to stop all live streams that might be in progress
    """
    stop_live_object_stream()
    stop_live_motion_stream()


def stop_live_object_stream():
    """
    Stop the live object detection stream
    """
    global live_object_detector_t
    if live_object_detector_t:
        live_object_detector_t.quit()
        live_object_detector_t.join()
        live_object_detector_t = None


def stop_live_motion_stream():
    """
    Stop the live motion detection stream
    """
    global live_motion_detector_t
    if live_motion_detector_t:
        live_motion_detector_t.quit()
        live_motion_detector_t.join()
        live_motion_detector_t = None
