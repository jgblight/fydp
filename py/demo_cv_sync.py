#!/usr/bin/env python
import freenect
import numpy as np
import cv
import cv2
import frame_convert
import time

global color

def get_depth():
    return frame_convert.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert.video_cv(freenect.sync_get_video()[0])

class colorFilter:
    def __init__(self):
        self.lbound = 0
        self.ubound = 0
        self.filter = False
        self.im = []

    def filterImage(self,im):
        imhsv = cv2.cvtColor(im,cv.CV_BGR2HSV)
        imfilter = cv2.inRange(imhsv,cv.Scalar(100,150,0),cv.Scalar(120,255,255))

        self.im = imhsv

        return im



    def onmouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            color = self.hsv[x,y,:]
            print color

history = []

cFilter = colorFilter()

cv2.namedWindow("Original",1)
cv2.setMouseCallback("Original", cFilter.onmouse)

while 1:
    try:
        #cv.ShowImage('Depth', get_depth())
        imbgr = get_video()
        cv2.imshow("Original",imbgr)

        imhsv = cv2.cvtColor(imbgr,cv.CV_BGR2HSV)
        imfilter = cv2.inRange(imhsv,cv.Scalar(100,150,0),cv.Scalar(120,255,255))

        imt = cFilter.filterImage(imbgr)

        cv2.imshow('Filtered', imfilter)


    except KeyboardInterrupt:
        break
    if cv.WaitKey(10) == 27:
        break
