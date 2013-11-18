#!/usr/bin/env python
import freenect
import numpy as np
import cv
import cv2
import cvblob
import frame_convert
import time

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

class colourFilter:

    def __init__(self,low,high):
        self.low = low
        self.high = high

    def getColourHull(self,imbgr):
        imycrcb = cv2.cvtColor(imbgr,cv.CV_BGR2YCrCb)
        imfilter = cv2.inRange(imycrcb,self.low,self.high)
        mask = np.zeros(np.add(imfilter.shape,[2,2]),dtype="uint8")
        filled = np.copy(imfilter)


        imfilter = cv2.medianBlur(imfilter,7)
        imfilter = cv2.medianBlur(imfilter,5)
        imfilter = cv2.medianBlur(imfilter,3)
        cv2.imshow("Blargh",imfilter)

        contours, hierarchy = cv2.findContours(imfilter,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours):
            mergeContour = contours[0]
            for i in contours[1:]:
                mergeContour = np.concatenate((mergeContour,i))

            return cv2.convexHull(mergeContour)
        else:
            return np.array([])

def getHuMoments(hull):
    if len(hull):
        moments = cv2.moments(hull)
        hu = cv2.HuMoments(moments)
        feature = []
        for i in hu:
            feature.append(i[0])
    else:
        feature = np.zeros(7)
    return feature

def getZernickeMoments(hull):
    pass

def getFeatureVector(hull):
    return getHuMoments(hull)


