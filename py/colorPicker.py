#!/usr/bin/env python
import freenect
import numpy as np
import cv
import cv2
import frame_convert
import time
import math

imbgr = 0
hsv = []
yuv = []

def get_depth():
    return frame_convert.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert.video_cv(freenect.sync_get_video()[0])

def onmouse(event, x, y, flags, param):
    global imbgr,hsv,yuv
    if event == cv2.EVENT_LBUTTONDOWN:
        colorHSV = cv2.cvtColor(imbgr,cv.CV_BGR2HSV)[y,x,:]
        colorYCrCb = cv2.cvtColor(imbgr,cv.CV_BGR2YCrCb)[y,x,:]
        hsv.append(colorHSV)
        yuv.append(colorYCrCb)
        print str(colorHSV) + "     " + str(colorYCrCb)

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

    n = 0.0
    hs1 = 0.0
    hs2 = 0.0
    ss1 = 0.0
    ss2 = 0.0
    vs1 = 0.0
    vs2 = 0.0
    for c in hsv:
        n+=1
        hs1+=float(c[0])
        hs2+=float(c[0])*float(c[0])
        ss1+=float(c[1])
        ss2+=float(c[1])*float(c[1])
        vs1+=float(c[2])
        vs2+=float(c[2])*float(c[2])

    hmn = hs1/n
    hstd = np.sqrt((n*hs2 - hs1*hs1)/(n*(n-1)))

    smn = ss1/n
    sstd = np.sqrt((n*ss2 - ss1*ss1)/(n*(n-1)))

    vmn = vs1/n
    vstd = np.sqrt((n*vs2 - vs1*vs1)/(n*(n-1)))

    print "N = " + str(n)
    print "HSV"
    print "mn = " + str([hmn,smn,vmn])
    print "std = " + str([hstd,sstd,vstd])
    print ' '

    n = 0.0
    hs1 = 0.0
    hs2 = 0.0
    ss1 = 0.0
    ss2 = 0.0
    vs1 = 0.0
    vs2 = 0.0
    for c in yuv:
        n+=1
        hs1+=float(c[0])
        hs2+=float(c[0])*float(c[0])
        ss1+=float(c[1])
        ss2+=float(c[1])*float(c[1])
        vs1+=float(c[2])
        vs2+=float(c[2])*float(c[2])

    hmn = hs1/n
    hstd = np.sqrt((n*hs2 - hs1*hs1)/(n*(n-1)))

    smn = ss1/n
    sstd = np.sqrt((n*ss2 - ss1*ss1)/(n*(n-1)))

    vmn = vs1/n
    vstd = np.sqrt((n*vs2 - vs1*vs1)/(n*(n-1)))

    print "YCrCb"
    print "mn = " + str([hmn,smn,vmn])
    print "std = " + str([hstd,sstd,vstd])
    print ' '

