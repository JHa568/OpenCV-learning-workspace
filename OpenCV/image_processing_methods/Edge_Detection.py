import cv2 as cv

# Edge detection is outlining the edges in the image
# such that the edges is the steep change in colour gradient

class Edge_Detection():

    # OpenCV
    def canny_detect(self, img):
        cpy_read = cv.imread(img)

        # @param - image, high bound, low bound
        # Doc for Canny Algorithm is:
        # https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
        edge_img = cv.Canny(img, 200, 100)
        return edge_img

    # Sci-kit image
