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
        green = extract.colourFilter(tuple(low),tuple(high))

        low = [ float(x) for x in reader.next()]
        high = [ float(x) for x in reader.next()]
        blue = extract.colourFilter(tuple(low),tuple(high))

    while 1:
        try:
            #cv.ShowImage('Depth', get_depth())
            imbgr = extract.get_video()
            #imbgr = imbgr[5:-5,5:-5]
            
            cv2.imshow("Original",imbgr)

            greenhull = green.getColourHull(imbgr)

            bluecentroid = blue.getColourCentroid(imbgr)
            
            imgray = cv2.cvtColor(imbgr,cv.CV_BGR2GRAY)

            cv2.drawContours(imgray,[greenhull],-1,(255,0,0),2)

            cv2.circle(imgray,tuple([int(bluecentroid[0]),int(bluecentroid[1])]),3,(255,0,0),3)

            cv2.imshow('Filtered', imgray)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break