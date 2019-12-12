"""
This package contains constant values used across all different use cases

@author: nidragedd
"""
import os

FONT_COLOR_RED = (0, 0, 255)
FONT_COLOR_GREEN = (0, 255, 0)

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
CONFIDENCE_THRESHOLD = 0.2
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

MODELS_PASCALVOC12 = os.path.join(MODELS_DIR, 'models', 'PascalVOC2012')
MODELS_COCO = os.path.join(MODELS_DIR, 'models', 'COCO')
MODELS_ILSVRC16 = os.path.join(MODELS_DIR, 'models', 'ILSVRC2016')

MOBILENET_300_PASCALVOC12 = os.path.join(MODELS_PASCALVOC12, 'MobileNet_SSD_300x300')
VGG_300_COCO = os.path.join(MODELS_COCO, 'VGGNet_SSD_300x300')
VGG_300_ILSVRC16 = os.path.join(MODELS_ILSVRC16, 'VGGNet_SSD_300x300')

PASCALVOC2012_CLASSES = os.path.join(MODELS_PASCALVOC12, 'voc2012.names')
COCO_CLASSES = os.path.join(MODELS_COCO, 'mscoco_labels.names')
ILSVRC16_CLASSES = os.path.join(MODELS_ILSVRC16, 'ilsvrc16_labels.names')
