from flask import render_template, request
from flask import Response
from imutils.video import VideoStream
import os

from src import app
from src.utils import utils
from src.streaming import concurrent
from src.streaming.livestreaming import LiveStreamer
from src.streaming.livereader import LiveReader

live_reader_t = None


@app.route('/')
def index():
    """
    Display the homepage. HTML file has to be placed within a 'templates' folder near by this module file as per
    Flask conventions
    """
    return render_template('master.html')


@app.route('/live')
def livestream():
    """
    Display live motion detection page.
    """
    global live_reader_t
    if not live_reader_t:
        live_reader_t = LiveReader(32, VideoStream(src=0), concurrent.lock, concurrent.lock_gray)
        live_reader_t.start()
    return render_template('livestream.html')


@app.route('/img_carousel', methods=['GET', 'POST'])
def obj_detection():
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'static', 'data', 'images'))
    img_list = utils.get_list_img_files(img_dir)
    car_item = request.args.get('car_item') if request.args.get('car_item') else 0
    return render_template('image_carousel.html', firstid=(img_list[0]), img_list=(img_list[1:]), car_item=car_item)


@app.route('/do_img_detection', methods=['POST'])
def launch_obj_detection():
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'static', 'data', 'images'))
    img_id = request.form['picked_img_id']
    img_url = request.form['picked_img_url']
    return render_template('img_ssd_models.html', img_id=img_id, img_url=img_url)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/video_feed')
def video_feed():
    """
    Use a generator to return the response generated along with the specific media type (mime type)
    :return: a Flask Response element wrapping a frame containing image and bounding box around detected motion
    """
    t = LiveStreamer(concurrent.lock)
    return Response((t.run()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_delta')
def video_feed_delta():
    """
    Use a generator to return the response generated along with the specific media type (mime type)
    :return: a Flask Response element wrapping a frame containing image and bounding box around detected motion
    """
    t = LiveStreamer(concurrent.lock_gray, use_gray=True)
    return Response((t.run()), mimetype='multipart/x-mixed-replace; boundary=frame')
