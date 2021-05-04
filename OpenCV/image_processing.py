import cv2 as cv
import sys

#import tkinter as tk
from image_processing_methods.Filters import Filters
from image_processing_methods.Edge_Detection import Edge_Detection
from image_processing_methods.Morphs import Morphology
from image_processing_methods.Corner_Detection import Corner_Detection

'''
Low-level image processing
'''
window_height = 600
window_width = 800

def image_thresholding(img):
        # These are the filtering values
        # They need to be tweaked via trial and error to find the
        # best bounds to detect the image
        H_h = 255
        H_s = 255
        H_v = 255
        L_h = 0
        L_s = 0
        L_v = 255

        processor_type = Filters()
        cvted_img, processed_img, original = processor_type.isolating_object(img,
                                                                    H_h,
                                                                    H_s,
                                                                    H_s,
                                                                    L_h,
                                                                    L_s,
                                                                    L_v)
        #cv.imshow("hsv img", cvted_img)
        #cv.imshow("original image", original)
        return cvted_img, processed_img, original

def image_morphing_dialate(img):
    morph = Morphology()
    dialated_img = morph.dialate(img)
    return dialated_img

def image_morphing_erode(img):
    morph = Morphology()
    eroded_img = morph.erode(img)
    return eroded_img

def setup() -> None:
    img = cv.imread("block.jpg")
    img_resized = cv.resize(img, (window_height, window_height))

    if img is None:
        sys.exit("Could not read the image")

    ####
    cvted_img, processed_img, original = image_thresholding(img_resized)
    dialate = image_morphing_dialate(processed_img)
    erode = image_morphing_erode(processed_img)
    ####
    cv.imshow("Processed", processed_img)
    #cv.imshow("dialated (5, 5)", dialate)
    #cv.imshow("eroded (5, 5)", erode)
    k = cv.waitKey(0)

if "__main__" == __name__:
    setup()
