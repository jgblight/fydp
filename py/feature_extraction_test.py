import numpy as np
import featureExtraction as extract
import cv
import cv2
import sys
import csv


if __name__ == "__main__":

    with open(sys.argv[1]) as csvfile:
        reader = csv.reader(csvfile)
        low = [ float(x) for x in reader.next()]
        high = [ float(x) for x in reader.next()]
        print low
        print high

    green = extract.colourFilter(tuple(low),tuple(high))

    while 1:
        try:
            #cv.ShowImage('Depth', get_depth())
            imbgr = extract.get_video()
            #imbgr = imbgr[5:-5,5:-5]
            
            cv2.imshow("Original",imbgr)

            hull = green.getColourHull(imbgr)
            
            imgray = cv2.cvtColor(imbgr,cv.CV_BGR2GRAY)

            cv2.drawContours(imgray,[hull],-1,(255,0,0),2)


            #cvFilter = toCVMat(imfilter,1)
            #imlabel = cv.CreateImage((imfilter.shape[1],imfilter.shape[0]),cvblob.IPL_DEPTH_LABEL, 1)
            #blobs = cvblob.Blobs()
            #result = cvblob.Label(cvFilter,imlabel,blobs)
            #print result
            #print len(blobs.keys())


                #cv2.rectangle(imgray,(leftmost,topmost),(rightmost,bottommost),(255,255,255))

            print extract.getFeatureVector(hull)
            cv2.imshow('Filtered', imgray)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break