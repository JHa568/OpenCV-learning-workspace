import cv2 as cv
import numpy as np

# There are many morphs but these are the main morphs that many people use

class Morphology():
    # OpenCV
    # Adding more
    def dialate(self, img, kernal_size=(5,5)):
        kernal = np.ones(kernal_size, np.uint8)
        dialate = cv.dilate(img, kernal, iterations=1)
        return dialate

    # Removing noise
    def erode(self, img, kernal_size=(5,5)):
        kernal = np.ones(kernal_size, np.uint8)
        erosion = cv.erode(img, kernal, iterations=1)
        return erosion

    def opening(self, img, kernal_size=(1,1)):
        kernal = np.ones(kernal_size, np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return opening

    def closing(self, img, kernal_size=(1,1)):
        kernal = np.ones(kernal_size, np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return closing



    # Sci-kit image
