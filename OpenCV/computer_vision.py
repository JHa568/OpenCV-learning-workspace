"""
This is the computer vision task in the medical field and seeing it how it can be utilised
"""

import cv2 as cv


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = frame
        # Display the resulting frame
        cv.imshow('frame', gray)

        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if "__main__" == __name__:
    print("Beginning of program")
    main()
    print("End of program")
