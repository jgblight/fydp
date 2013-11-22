#!/usr/bin/env python
import freenect
import numpy as np
import cv
import cv2
import frame_convert
import time
import math
import csv

def get_depth():
    return frame_convert.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert.video_cv(freenect.sync_get_video()[0])


def onmouse(event, x, y, flags, param):
    global imbgr,hsv,yuv
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.cv.SetImageROI(imbgr,  cvRect(x, y, x + 25, y + 25) )

	colorHSV = cv2.cvtColor(imbgr,cv.CV_BGR2HSV)[y,x,:]
        colorYCrCb = cv2.cvtColor(imbgr,cv.CV_BGR2YCrCb)[y,x,:]

def show(filename):
    global imbgr,hsv,yuv
    cv2.namedWindow("Original",1)
    cv2.setMouseCallback("Original", onmouse)
    hsv = []
    yuv = []

    while 1:
        try:
            imbgr = cv2.imread(filename)
            cv2.imshow("Original",imbgr)

            #imhsv = cv2.cvtColor(imbgr,cv.CV_BGR2HSV)
            #imfilter = cv2.inRange(imhsv,cv.Scalar(0,0,0),cv.Scalar(255,10,255))

            #cv2.imshow('Filtered', imfilter)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break
