import numpy as np
import sys
import os
import cv
import cv2
import featureExtraction as extract
from autocalibrate import autocalibrate

def get_mean(imbgr,f):
    imbw = f.markers['right'].inRange(imbgr)
    count = np.sum(imbw/255)

    imyuv = cv2.cvtColor(imbgr,cv.CV_BGR2YCrCb)
    imfilter = cv2.bitwise_and(imyuv,imyuv,mask=imbw)
    mean = np.sum(np.sum(imfilter,0),0)
    mean = mean/float(count)
    return mean

def get_mask(imbgr,mean):
    imyuv = cv2.cvtColor(imbgr,cv.CV_BGR2YCrCb)
    D = np.max(np.max(np.absolute(imyuv-mean),0),0)

    S = np.sqrt(np.sum(np.square(np.absolute(imyuv-mean)/D),2))
    S = (S/np.max(S))
    h,i = np.histogram(S,256)
    H = np.convolve(h,np.ones(10)*1/10.0,'same')
    
    p = 0
    while H[p] >= np.max(H[:p+1]) and p < 255:
        p += 1
    p -= 1


    m = p
    while H[m] <= np.min(H[p:m+1]) and m < 255:
        m += 1
    m -= 1

    mask = np.where(S <= i[m],np.ones(S.shape),np.zeros(S.shape))*255
    return mask


def main():
    
    #autocalibrate(sys.argv[1])
    f = extract.FeatureExtractor(sys.argv[1])
    f.setStartPoint()
    imbgr = np.array(extract.get_video())
    mean  = get_mean(imbgr,f)
    while 1:
        try:
            imbgr = np.array(extract.get_video())
            mask = get_mask(imbgr,mean)

            print np.max(mask)
            print np.min(mask)
            mask = mask.astype('uint8')
            print np.max(mask)
            print np.min(mask)
            mask = f.markers['right'].blobSmoothing(mask)
            cv2.imshow('Demo', mask)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break


if __name__ == "__main__":
    main()
