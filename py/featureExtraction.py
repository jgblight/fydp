#!/usr/bin/env python
import freenect
import numpy as np
import cv
import cv2

import frame_convert
import cmath
import math
import csv

import hashlib
import functools

class memoize(object):
    def __init__(self, func):
       self.func = func
       self.cache = {}
       self.oldhash = None
    def __call__(self, *args):
        if not len(args) == 2 or not isinstance(args[1],np.ndarray):
            return self.func(*args)
        newhash = hashlib.sha1(np.array(args[1]).view(np.uint8)).hexdigest()
        if not newhash == self.oldhash:
            self.cache = {}
            self.oldhash = newhash
        if args[0] not in self.cache:
            self.cache[args[0]] = self.func(*args)
        return self.cache[args[0]]
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

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

    @memoize
    def getColourHull(self,imbgr):
        contours = self.getColourContours(imbgr)
        if len(contours):
            mergeContour = contours[0]
            for i in contours[1:]:
                mergeContour = np.concatenate((mergeContour,i))

            return cv2.convexHull(mergeContour)
        else:
            return np.array([])

    def getStartingPoint(self,imbgr):
        contours = self.getColourContours(imbgr) 
        if len(contours):
            moments = cv2.moments(contours[0])
            if moments['m00'] > 0:
                return (int(moments['m10']/moments['m00']),int(moments['m01']/moments['m00'])) 
        return tuple([])

    @memoize
    def inRange(self,imbgr):
        imycrcb = cv2.cvtColor(imbgr,cv.CV_BGR2YCrCb)
        imfilter = cv2.inRange(imycrcb,self.low,self.high)
        return imfilter

    def blobSmoothing(self,immask):
        imfilter = cv2.medianBlur(immask,7)
        imfilter = cv2.medianBlur(imfilter,5)
        imfilter = cv2.medianBlur(imfilter,3)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

        imfilter = cv2.dilate(imfilter,kernel)
        imfilter = cv2.erode(imfilter,kernel)
        return imfilter

    @memoize
    def getColourContours(self,imbgr):
        imfilter = self.inRange(imbgr)
        imfilter = self.blobSmoothing(imfilter)

        contours, hierarchy = cv2.findContours(imfilter,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def getCombinedCentroid(self, imbgr, blue):
        imfilter = self.inRange(imbgr) + blue.inRange(imbgr)
        imfilter = self.blobSmoothing(imfilter)

        contours, hierarchy = cv2.findContours(imfilter,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        start = self.getStartingPoint(imbgr)

        if start:
            for cnt in contours:
                if cv2.pointPolygonTest(cnt,start,False) == 1:
                    break
            moments = cv2.moments(cnt)
            if moments['m00'] > 0:
                return (int(moments['m10']/moments['m00']),int(moments['m01']/moments['m00'])) 
        return tuple([])

class FeatureExtractor:

    def __init__(self,calibration):
        self.markers = {}
        with open(calibration) as csvfile:
            reader = csv.reader(csvfile)
            low = [ float(x) for x in reader.next()]
            high = [ float(x) for x in reader.next()]
            self.markers['right'] = colourFilter(tuple(low),tuple(high))

            low = [ float(x) for x in reader.next()]
            high = [ float(x) for x in reader.next()]
            self.markers['glove'] = colourFilter(tuple(low),tuple(high))

            low = [ float(x) for x in reader.next()]
            high = [ float(x) for x in reader.next()]
            self.markers['left'] = colourFilter(tuple(low),tuple(high))

    def getHandPosition(self,imbgr,imdepth,hand):
        centroid = self.markers[hand].getCombinedCentroid(imbgr, self.markers['glove'])
        if centroid:
            return np.append(centroid,imdepth[centroid[::-1]])
        return np.array([])

    def getCentralMoments(self,imbgr,hand):
        hull = self.markers[hand].getColourHull(imbgr)
        if len(hull):
            m = cv2.moments(hull)
            feature = [m['nu20'],m['nu11'],m['nu02'],m['nu30'],m['nu21'],m['nu12'],m['nu03']]
        else:
            feature = [0,0,0,0,0,0,0]
        return feature,hull

    def getHuMoments(self,imbgr,hand):
        hull = self.markers[hand].getColourHull(imbgr)
        if len(hull):
            m = cv2.moments(hull)
            hu = cv2.HuMoments(m)
            feature = []
            for i in hu:
                feature.append(i[0])
        else:
            feature = [0,0,0,0,0,0,0]
        return feature,hull
