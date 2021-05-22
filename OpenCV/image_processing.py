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

def setup() -> None:
    img = cv.imread("block.jpg")
    img_resized = cv.resize(img, (window_height, window_height))

    if img is None:
        sys.exit("Could not read the image")

    # ed = Edge_Detection()
    # edge = ed.canny_detect(img_resized)
    # cv.imshow('edge', edge)
    # corner = Corner_Detection()
    # cd = corner.detect(img_resized)
    #
    # img_resized[cd>0.009*cd.max()]=[0,0,255]
    #
    # cv.imshow('test', img_resized)

    c, p, o = image_thresholding(img_resized)
    cv.imshow("Processed", p)
    '''
    morph = Morphology()
    cvted_img, processed_img, original = image_thresholding(img_resized)
    opening = morph.opening(processed_img)
    closing = morph.closing(opening)
    '''
    ####
    '''
    cv.imshow("Processed", processed_img)
    cv.imshow("opening", opening)
    cv.imshow("closing", closing)
    '''

    k = cv.waitKey(0)

if "__main__" == __name__:
    setup()
