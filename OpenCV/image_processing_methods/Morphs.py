import cv2 as cv
import numpy as np

# There are many morphs but these are the main morphs that many people use

class Morphology():
    # OpenCV
    # Adding more
    def dialate(self, img, kernal_size=(5,5)):
        kernel = np.ones(kernal_size, np.uint8)
        dialate = cv.dilate(img, kernel, iterations=1)
        return dialate

    # Removing noise
    def erode(self, img, kernel_size=(5,5)):
        kernel = np.ones(kernel_size, np.uint8)
        erosion = cv.erode(img, kernel, iterations=1)
        return erosion

    # Enhancing/dialating the white object
    def opening(self, img, kernal_size=(5,5)):
        kernel = np.ones(kernal_size, np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        return opening

    def closing(self, img, kernal_size=(5,5)):
        kernel = np.ones(kernal_size, np.uint8)
        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        return closing

<<<<<<< HEAD
=======

    def test(): -> None
        return None
>>>>>>> 87cc6619e4b274b67de7a219ddf76ceed56c1461
    # Sci-kit image
