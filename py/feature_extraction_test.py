import numpy as np
import featureExtraction as extract
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import cv
import cv2
import sys


if __name__ == "__main__":

    rightPast = deque([])
    leftPast = deque([])
    f = extract.FeatureExtractor(sys.argv[1])


    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    while 1:
        try:
            #cv.ShowImage('Depth', get_depth())
            imbgr = extract.get_video()
            imdepth = extract.get_depth()
            
            cv2.imshow("Original",imbgr)

            greenmoments,greenhull = f.getCentralMoments(imbgr,'right')
            redmoments,redhull = f.getCentralMoments(imbgr,'left')

            imgray = cv2.cvtColor(imbgr,cv.CV_BGR2GRAY)
            imgray2 = cv2.cvtColor(imgray,cv.CV_GRAY2BGR)

            cv2.drawContours(imgray2,[greenhull],-1,(255,0,0),2)
            cv2.drawContours(imgray2,[redhull],-1,(255,0,0),2)

            leftcentroid = f.getHandPosition(imbgr,imdepth,'left')
            rightcentroid = f.getHandPosition(imbgr,imdepth,'right')
        
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

            if len(lefthand):
                ax.scatter(lefthand[0],lefthand[1],lefthand[2],c='r')
            if len(righthand):
                ax.scatter(righthand[0],righthand[1],righthand[2],c='g')
            plt.draw()

            print str(lefthand) + "    " + str(righthand)

            cv2.imshow('Filtered', imgray2)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break
