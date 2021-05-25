import cv2 as cv
import sys

from image_processing_methods.Filters import Filters

'''
Low-level image processing
'''
window_height = 600
window_width = 800

def setup() -> None:
    if len(sys.argv) > 2 or len(sys.argv) <= 1:
        print("invalid arguments")

    else:

        img = cv.imread(sys.argv[1])
        img_resized = cv.resize(img, (window_height, window_height))

        if img is None:
            sys.exit("Could not read the image")

        filter = Filters()


        bi = filter.bilateral_filter(img_resized)
        guass = filter.guassian_filter(img_resized)
        avg = filter.averaging(img_resized)
        med_blur = filter.median_blur(img_resized)

        ####

        cv.imshow("median_blur", med_blur)
        cv.imshow("bilateral_filter", bi)
        cv.imshow("guassian_filter", guass)
        cv.imshow("averaging", avg)
        cv.imshow("original", img_resized)

        k = cv.waitKey(0)

if "__main__" == __name__:
    setup()
