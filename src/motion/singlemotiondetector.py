"""
Based on this blog post: https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/
"""
import numpy as np
import imutils
import cv2


class SingleMotionDetector:
    """'
    We will use this class to perform background subtraction and motion detection.

    Most background subtraction algorithms work by:
            1. Accumulating the weighted average of the previous N frames
            2. Taking the current frame and subtracting it from the weighted average of frames
            3. Thresholding the output of the subtraction to highlight the regions with substantial differences in pixel
             values (“white” for foreground and “black” for background)
            4. Applying basic image processing techniques such as erosions and dilations to remove noise
            5. Utilizing contour detection to extract the regions containing motion

    This is a “single motion detector” as the algorithm itself is only interested in finding the single, largest region
    of motion. We could extend this method to handle multiple regions of motion as well.
    """

    def __init__(self, accum_weight=0.5):
        """
        Constructor with a default value for accumulated weight
        The larger accum_weight is, the less the background will be factored in when accumulating the weighted average.
        Conversely, the smaller accum_weight is, the more the background bg will be considered when computing the
        average.
        Setting accum_weight=0.5 weights both the background and foreground evenly
        :param accum_weight: (float) factor used to our accumulated weighted average
        """
        self.accum_weight = accum_weight
        self.background_model = None

    def update(self, image):
        """
        Take an input frame and compute the weighted average: on first call, store the background frame.
        Otherwise, compute the weighted average between the input frame and the existing background, using our
        corresponding accum_weight factor.
        :param image: the frame to process
        """
        if self.background_model is None:
            self.background_model = image.copy().astype('float')
            return
        cv2.accumulateWeighted(image, self.background_model, self.accum_weight)

    def detect(self, image, threshold=25):
        """
        Given our input image we compute the absolute difference between the image and the background. Then all pixels
        that have a difference > threshold are set to 255 (i.e white), otherwise they are set to 0 (black).
        Erosions and dilations are performed to remove noise and small, localized areas of motion that would otherwise
        be considered false-positives (likely due to reflections or rapid changes in light).
        Then we find contours and build a bounding box with maximum width and height (i.e a single wide box that wraps
        all motions)
        :param image: The input frame/image that motion detection will be applied to. As per documentation, source
        image should be a grayscale image.
        :param threshold: The threshold value used to mark a particular pixel as “motion” or not.
        :return: (tuple) 1st element is thresholded image , 2nd is another tuple of 4 box coordinates
        """
        # Compute absolute difference between background model and the image passed in, then threshold the delta image
        delta = cv2.absdiff(self.background_model.astype('uint8'), image)
        thresh = cv2.threshold(delta, threshold, 255, cv2.THRESH_BINARY)[1]

        # Perform a series of erosions and dilations to remove small blobs
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the threshold image and initialize the minimum and maximum bounding box regions for motion
        img_contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = imutils.grab_contours(img_contours)
        minX, minY = np.inf, np.inf
        maxX, maxY = -np.inf, -np.inf

        # If no contours found there is no motion, we stop
        if len(img_contours) == 0:
            return None
        else:
            # Compute bounding box of each contour and use it to update the minimum and maximum bounding box regions
            for c in img_contours:
                x, y, w, h = cv2.boundingRect(c)
                minX, minY = min(minX, x), min(minY, y)
                maxX, maxY = max(maxX, x + w), max(maxY, y + h)

            return thresh, (minX, minY, maxX, maxY), img_contours
