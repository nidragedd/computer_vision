"""
This package contains constant values used across all different use cases

@author: nidragedd
"""
FONT_COLOR_RED = (0, 0, 255)
FONT_COLOR_GREEN = (0, 255, 0)
FONT_COLOR_YELLOW = (0, 255, 255)

####################################
#   LIVE STREAMING
####################################
VIDEOSTREAM_WARMUP = 1.0
RUNNING_STREAM_STATUS = "RUNNING"
STREAM_MOTION = "STREAM_MOTION"
STREAM_OBJECT = "STREAM_OBJECT"
FRAME_RESIZED_VALUE = 600
LIVE_MOTION_DETECTION_MIN_AREA = 300

####################################
#   OBJECTS DETECTION
####################################
DETECT_CONFIG_JSON_KEY = "detection"
SSD_MODEL_TYPE = "SSD"
CONFIDENCE_THRESHOLD = 0.2

####################################
#   HUMAN POSE ESTIMATION
####################################
POSE_CONFIG_JSON_KEY = "pose_estimation"
POSE_CONFIDENCE_THRESHOLD = 0.2
