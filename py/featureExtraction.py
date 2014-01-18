#!/usr/bin/env python
import freenect
import numpy as np
import cv
import cv2

import frame_convert
from scipy.misc import factorial as fac
import cmath
import math

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
        contours = self.getColourContours(imbgr)
        if len(contours):
            mergeContour = contours[0]
            for i in contours[1:]:
                mergeContour = np.concatenate((mergeContour,i))

            return cv2.convexHull(mergeContour)
        else:
            return np.array([])

    def getColourCentroid(self,imbgr):
        contours = self.getColourContours(imbgr)
        if len(contours):
            centroid_numerator = np.array([0,0])
            centroid_denominator = 0.0
            for cnt in contours:
                moments = cv2.moments(cnt)
                centroid_denominator += moments['m00'] 
                centroid_numerator += [moments['m10'],moments['m01']] 
            return centroid_numerator / centroid_denominator
        else:
            return np.array([])

    def getColourContours(self,imbgr):
        imycrcb = cv2.cvtColor(imbgr,cv.CV_BGR2YCrCb)
        imfilter = cv2.inRange(imycrcb,self.low,self.high)
        mask = np.zeros(np.add(imfilter.shape,[2,2]),dtype="uint8")
        filled = np.copy(imfilter)


        imfilter = cv2.medianBlur(imfilter,7)
        imfilter = cv2.medianBlur(imfilter,5)
        imfilter = cv2.medianBlur(imfilter,3)

        contours, hierarchy = cv2.findContours(imfilter,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours

#    def getCombinedCentroid(self, imbgr, blue):
#	hand = self
#	while adjacent = true
# 		ajacentcheck = false
#		for point in hand
#			pointVicinity = [point+(0,1), point+(0,-1), point+(1,0), point+(-1,0)]
#			if any point in point vicinity is in blue
#				add point to hand
#				adjacentcheck = true	
#		if adjacentcheck == false
#			adjacent = false
#	combinedcentroid = centroid of hand		 				
#	return combinedcentroid

def getCentralMoments(hull):
    if len(hull):
        m = cv2.moments(hull)
        feature = [m['nu20'],m['nu11'],m['nu02'],m['nu30'],m['nu21'],m['nu12'],m['nu03']]
    else:
        feature = [0,0,0,0,0,0,0]
    return feature

def getHuMoments(hull):
    if len(hull):
        m = cv2.moments(hull)
        hu = cv2.HuMoments(m)
        feature = []
        for i in hu:
            feature.append(i[0])
    else:
        feature = [0,0,0,0,0,0,0]
    return feature

def ZernikeMom(n,l,image,xc,yc,N):
    N = float(N)
    n = float(n)
    l = float(l)
    xran,yran,_ = image.shape
    A = complex(0,0)

    coeff = []
    for m in range(int((n-l)/2 + 1)):
        if m%2 == 0:
            c = 1
        else:
            c = -1
        den = (fac(m)*fac((n-2*m-l)/2.)*fac((n-2*m+l)/2.))
        coeff.append(c*fac(n-m)/den)

    for x in range(xran):
        xn = (x-xc)/N
        for y in range(yran):
            yn = (y-yc)/N
            zbf = complex(0,0)
            for m in range(int((n-l)/2 + 1)):
                rho = xn*xn + yn*yn
                if rho <= 1:
                    if xn:
                        theta = math.atan(yn/xn)
                    else:
                        theta = math.pi/2
                    zbf += coeff[m]* (rho**(n/2-m))*cmath.exp(complex(0,l*theta))
            A += image[x,y] * zbf.conjugate()
    return abs(A * (n+1)/np.pi)

def getZernickeMoments(hull,maxorder):
    xc = 0
    yc = 0
    N = 0
    moments = []
    if len(hull):
        leftmost = hull[hull[:,:,0].argmin()][0][0]
        rightmost = hull[hull[:,:,0].argmax()][0][0]
        topmost = hull[hull[:,:,1].argmin()][0][1]
        bottommost = hull[hull[:,:,1].argmax()][0][1]

        M = cv2.moments(hull)
        xc = int(M['m10']/M['m00']) - leftmost
        yc = int(M['m01']/M['m00']) - topmost

        N = max(rightmost - leftmost,bottommost - topmost)

        imfilled = np.zeros((rightmost+1,bottommost+1,1))
        cv2.drawContours(imfilled,[hull],-1,(255,0,0),-1)
        imfilled = imfilled[leftmost:,topmost:]

    for n in range(maxorder+1):
        for l in range(n+1):
            if (n-l)%2 == 0:
                if N:
                    moments.append(ZernikeMom(n,l,imfilled,xc,yc,N))
                    
                else:
                    moments.append(0)
    return moments

def getFeatureVector(hull,featureSet):
    feature = []
    if 'central' in featureSet:
        feature += getCentralMoments(hull)
    if 'hu' in featureSet:
        feature += getHuMoments(hull)
    if 'zernike' in featureSet:
        feature += getZernickeMoments(hull,3)
    return feature


