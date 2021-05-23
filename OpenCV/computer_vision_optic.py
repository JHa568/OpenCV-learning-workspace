import numpy as np
import cv2 as cv
from low_level_vision.optical_flow import optical_flow

# cv.samples.findFile("security.mp4")
cap = cv.VideoCapture(0)

# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

hsv = np.zeros_like(old_frame)
hsv[...,1] = 255

optical_flow = optical_flow()
while True:
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculate optical flow
    # Dense flow -> optical_flow.Denseflow(old_gray, frame_gray, hsv)#
    # Sparse flow -> optical_flow.Sparseflow(frame, old_frame, old_gray, frame_gray)#
    flow_img = optical_flow.Denseflow(old_gray, frame_gray, hsv)#

    ### FRAMES ###
    cv.imshow('optical_flow', flow_img)
    cv.imshow('frame', frame_gray)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
