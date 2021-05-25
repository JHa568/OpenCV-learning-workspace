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

        return cvted_img, processed_img, original

def setup() -> None:
    img = cv.imread("block.jpg")
    img_resized = cv.resize(img, (window_height, window_height))

    if img is None:
        sys.exit("Could not read the image")

    morph = Morphology()
    ed = Edge_Detection()
    corner = Corner_Detection()
    cvted_img, processed_img, original = image_thresholding(img_resized)

    cd = corner.detect(img_resized)
    edge = ed.canny_detect(img_resized)
    opening = morph.opening(processed_img)
    closing = morph.closing(opening)
    erode = morph.erode(processed_img)
    dialate = morph.dialate(processed_img)

    ####
    cv.imshow("original", img_resized)
    cv.imshow("Processed", processed_img)
    cv.imshow("HSV image", cvted_img)
    cv.imshow("Erode", erode)
    cv.imshow("Dialate", dialate)
    cv.imshow("Opening", opening)
    cv.imshow("Closing", closing)
    cv.imshow("Edge", edge)
    cv.imshow("original", original)

    img_resized[cd>0.009*cd.max()]=[0,0,255] # red dots
    cv.imshow('Corner detect', img_resized)

    k = cv.waitKey(0)

if "__main__" == __name__:
    setup()
