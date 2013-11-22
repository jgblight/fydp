#!/usr/bin/env python
import freenect
import numpy as np
import cv
import cv2
import cvblob
import frame_convert
import time

imbgr = 0

def get_depth():
    return frame_convert.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert.video_cv(freenect.sync_get_video()[0])

def toCVMat(im,channels):
    image = cv.CreateImage((im.shape[1], im.shape[0]),
                                 cv.IPL_DEPTH_8U,
                                 channels)
    cv.SetData(image, im.tostring(),
               im.dtype.itemsize * channels * im.shape[1])
    return image


while 1:
    try:
        #cv.ShowImage('Depth', get_depth())
        imbgr = get_video()
        cv2.imshow("Original",imbgr)

        imhsv = cv2.cvtColor(imbgr,cv.CV_BGR2HSV)
        imfilter = cv2.inRange(imhsv,cv.Scalar(35,80,80),cv.Scalar(50,255,255))
        mask = np.zeros(np.add(imfilter.shape,[2,2]),dtype="uint8")
        filled = np.copy(imfilter)
        cv2.floodFill(filled,mask,(0,0),(255,255,255))
        imfilter = imfilter + 255 - filled
        imfilter = cv2.medianBlur(imfilter,3)

        contours, hierarchy = cv2.findContours(imfilter,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(imfilter,contours,-1,(255,0,0),2)


        #cvFilter = toCVMat(imfilter,1)
        #imlabel = cv.CreateImage((imfilter.shape[1],imfilter.shape[0]),cvblob.IPL_DEPTH_LABEL, 1)
        #blobs = cvblob.Blobs()
        #result = cvblob.Label(cvFilter,imlabel,blobs)
        #print result
        #print len(blobs.keys())

        cv2.imshow('Filtered', imfilter)

    except KeyboardInterrupt:
        break
    if cv.WaitKey(10) == 27:
        break
