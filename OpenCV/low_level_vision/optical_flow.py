import cv2 as cv
import numpy as np

'''
For a low level vision task.
Can be used for detecting the motion in the camera.
'''

class optical_flow():
    def __init__(self):
        self.good_new = []
        self.good_old = []

    def Denseflow(self, prev_img, next_img, hsv, flow=None, pry_scale=0.5, lvl=3, winsize=15,
                                                                           iterations=3,
                                                                           poly_n=5,
                                                                           poly_sigma=1.2,
                                                                           flags=0):

        # This will show the previous movement of
        flow = cv.calcOpticalFlowFarneback(prev_img, next_img, flow, pry_scale, lvl, winsize, iterations, poly_n, poly_sigma, flags)
        # Changes the flow to polar coordinates
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        # Makes the hue value ang*180/np.pi/2. what angle did the frame update to
        hsv[...,0] = ang*180/np.pi/2
        # Makes the value is normalised.
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        return bgr

    def Sparseflow(self, frame, old_frame, old_gray, frame_gray):
            # Create some random colors
            color = np.random.randint(0,255,(100,3)) # tracker markers
            p0 = cv.goodFeaturesToTrack(old_gray, mask = None, maxCorners = 100,
                                                               qualityLevel = 0.3,
                                                               minDistance = 7,
                                                               blockSize = 7 )
            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)

            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize  = (15,15),
                                                                                  maxLevel = 2,
                                                                                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
            # Select good points
            if p1 is not None:
                self.good_new = p1[st==1]
                self.good_old = p0[st==1]

            # draw the markers
            for i,(new,old) in enumerate(zip(self.good_new, self.good_old)):
                a,b = new.ravel()
                frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)

            img = cv.add(frame, mask)
            p0 = self.good_new.reshape(-1,1,2)
            return img
