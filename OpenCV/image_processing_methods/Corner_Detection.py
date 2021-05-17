import cv2 as cv
import numpy as np

class Corner_Detection():

    # OpenCV
    def detect(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        gray = np.float32(gray)
        print("Gray -> ", np.asarray(gray).shape)
        corner = cv.cornerHarris(gray,2,9,0.04)
        return corner

    # Sci-kit image
