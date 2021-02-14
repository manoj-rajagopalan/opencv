# Read video files in Python

import cv2 as cv
import sys
from tqdm import tqdm

print('OpenCV version: ', cv.__version__)
assert len(sys.argv) == 2, "Usage: {} <video file>".format(sys.argv[0])

video_file_reader = cv.VideoCapture(sys.argv[1]) # handles files also
for i in tqdm(range(200)):
    # print('Displaying frame ', (i+1))
    img = video_file_reader.read()  # returns (bool, ndarray of shape (height, width, channels))
    assert img[0]
    if(i == 0):
        print('frame dims = ', img[1].shape)
    # /if

    # draw red box
    img[1][100:300, 100, 2] = 255
    img[1][100:300, 500, 2] = 255
    img[1][100, 100:500, 2] = 255
    img[1][300, 100:500, 2] = 255

    # display
    cv.imshow('Frame'.format(i+1), img[1]) # syntax: window-name, data
    
    # required for flushing to screen, waits for given delay in ms (indefinitely if < 0)
    cv.waitKey(30) 
# for i

cv.destroyAllWindows()
