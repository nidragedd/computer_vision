"""
This is not my work and everything is based on this great blog post:
https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

This package contains methods to perform object detections

Acknowledgments: pyimagesearch blog for his great work
"""
import numpy as np
import cv2


def object_detection_from_image(ssd_model, image, threshold_confidence):
    """
    Given a trained network and an image, perform object detection.
    :param ssd_model: trained network loaded through opencv
    :param image: opencv image element
    :param threshold_confidence: (float) minimum level of confidence to detect elements
    :return: opencv image with bounding boxes around detected objects, their class and confidence
    """
    h, w = image.shape[:2]
    size = ssd_model.get_size()
    img_resized = cv2.resize(image, (size, size))
    blob = cv2.dnn.blobFromImage(img_resized, scalefactor=0.007843, size=(size, size), mean=127.5)
    print('Computing object detections...')
    ssd_model.get_net().setInput(blob)
    detections = ssd_model.get_net().forward()
    print('Found {} predictions'.format(detections.shape[2]))
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[(0, 0, i, 2)]
        if confidence > threshold_confidence:
            idx = int(detections[(0, 0, i, 1)])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x_min, y_min, x_max, y_max = box.astype('int')
            label = '{}: {:.2f}%'.format(ssd_model.get_label(idx), confidence * 100)
            print('Found {} in picture!'.format(label))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), ssd_model.get_color(idx), 2)
            y = y_min - 15 if y_min - 15 > 15 else y_min + 15
            cv2.putText(image, label, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ssd_model.get_color(idx), 2)

    return image
