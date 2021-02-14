# Read video files in Python

import cv2 as cv
import sys

print('OpenCV version: ', cv.__version__)
assert len(sys.argv) == 2, "Usage: {} <video file>".format(sys.argv[0])

video_file_reader = cv.VideoCapture(sys.argv[1]) # handles files also
for i in range(200):
    print('Displaying frame ', (i+1))
    img = video_file_reader.read()  # returns (bool, ndarray of shape (height, width, channels))
    assert img[0]
    cv.imshow('Frame'.format(i+1), img[1]) # syntax: window-name, data
    
    # required for presenting to screen, waits for given delay in ms (indefinitely if < 0)
    cv.waitKey(30) 
# for i

cv.destroyAllWindows()
