import numpy as np
import cv2 as cv
import sys
from low_level_vision.optical_flow import optical_flow


invalid_arg = False
# cv.samples.findFile("security.mp4")
if len(sys.argv) > 4 or len(sys.argv) <= 1:
    invalid_arg = True
    print("invalid arguments")

else:

    cap = cv.VideoCapture(cv.samples.findFile(sys.argv[1]))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    hsv = np.zeros_like(old_frame)
    hsv[...,1] = 255

    flow_type = sys.argv[2]


if "__main__" == __name__ and invalid_arg == False:

    optical_flow = optical_flow()

    while True:
        ret,frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculate optical flow
        # Dense flow -> optical_flow.Denseflow(old_gray, frame_gray, hsv)#
        # Sparse flow -> optical_flow.Sparseflow(frame, old_frame, old_gray, frame_gray)#
        if flow_type.lower() == "dense":
            flow_img = optical_flow.Denseflow(old_gray, frame_gray, hsv)#
        elif flow_type.lower() == "sparse":
            flow_img = optical_flow.Sparseflow(frame, old_frame, old_gray, frame_gray)#
        else:
            print("Unknown flow type")
            break

        ### FRAMES ###
        cv.imshow('optical_flow', flow_img)
        cv.imshow('frame', frame_gray)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
