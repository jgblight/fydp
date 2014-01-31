import numpy as np
import featureExtraction as extract
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import cv
import cv2
import sys
import csv


if __name__ == "__main__":

    rightPast = deque([])
    leftPast = deque([])
    with open(sys.argv[1]) as csvfile:
        reader = csv.reader(csvfile)
        low = [ float(x) for x in reader.next()]
        high = [ float(x) for x in reader.next()]
        green = extract.colourFilter(tuple(low),tuple(high))

        low = [ float(x) for x in reader.next()]
        high = [ float(x) for x in reader.next()]
        blue = extract.colourFilter(tuple(low),tuple(high))

        low = [ float(x) for x in reader.next()]
        high = [ float(x) for x in reader.next()]
        red = extract.colourFilter(tuple(low),tuple(high))


    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    while 1:
        try:
            #cv.ShowImage('Depth', get_depth())
            imbgr = extract.get_video()
            imdepth = extract.get_depth()
            
            cv2.imshow("Original",imbgr)

            greenhull = green.getColourHull(imbgr)
            redhull = red.getColourHull(imbgr)

            #rightcentroid = green.getCombinedCentroid(imbgr, blue,'Right')
            #leftcentroid = red.getCombinedCentroid(imbgr, blue,'Left')
            
            imgray = cv2.cvtColor(imbgr,cv.CV_BGR2GRAY)
            imgray2 = cv2.cvtColor(imgray,cv.CV_GRAY2BGR)

            cv2.drawContours(imgray2,[greenhull],-1,(255,0,0),2)
            cv2.drawContours(imgray2,[redhull],-1,(255,0,0),2)

            leftcentroid = extract.getCentroidPosition(imbgr,imdepth,red,blue)
            rightcentroid = extract.getCentroidPosition(imbgr,imdepth,green,blue)
	    
	    #begin centroid averaging filter
	    if len(leftPast) < 10:
		lefthand = leftcentroid
		leftPast.append(leftcentroid)
            else:
		_ = leftPast.popleft()
		leftPast.append(leftcentroid)
		lefthand = tuple(map(np.mean, zip(*leftPast)))

	    if len(rightPast) < 10:
		righthand = rightcentroid
		rightPast.append(rightcentroid)
            else:
		_ = rightPast.popleft()
		rightPast.append(rightcentroid)
		righthand = tuple(map(np.mean, zip(*rightPast)))
	    #end centroid averaging filter

            ax.scatter(lefthand[0],lefthand[1],lefthand[2],c='r')
            ax.scatter(righthand[0],righthand[1],righthand[2],c='g')
            plt.draw()

            print str(lefthand) + "    " + str(righthand)

            cv2.imshow('Filtered', imgray2)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break
