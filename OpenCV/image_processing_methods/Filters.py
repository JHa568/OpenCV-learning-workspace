import cv2 as cv
import numpy as np
'''
Filters are the backbone for image processing

They aid in obtaining required information from the image through the isolation
of bright or coloured objects as well as reducing the image noise.
'''
class Filters():

    # OpenCV
    def isolating_object(self, img, H_h, H_s, H_v, L_h, L_s, L_v) -> np.ndarray:
        # Isolate the object you want based on the color / brightness
        # This is known as image thresholding
        high_thresh = np.array([H_h, H_s, H_v],np.uint8)
        lower_thresh = np.array([L_h, L_s, L_v],np.uint8)

        # Depends on problem and image
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        img_isolated = cv.inRange(hsv_img, lower_thresh, high_thresh)
        return hsv_img, img_isolated, img

    '''
    Bluring/Smoothing the image is also a good way of reducing the noise
    since it is averaging pixels in the image.

    There are four blur tools in OpenCV
    '''
    # Remove minor noise in image while maintaining the sharpness of the edges
    def bilateral_filter(self, img):
        blur = cv.bilateralFilter(img, 9, 75, 75)
        return blur

    # Removes minor noise throughout the image
    def guassian_filter(self, img, kernal=(5,5)):
        #
        blur = cv.GaussianBlur(img, kernal, 0)
        return blur

    # Softens the image
    def averaging(self, img, kernal=(5,5)):
        blur = cv.blur(img, kernal)
        return blur

    # Very aggressive bluring
    def median_blur(self, img):
        blur = cv.medianBlur(img, 5)

    # Sci-kit image
