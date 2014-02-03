import numpy as np
import featureExtraction as extract
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import cv
import cv2
import sys
import time


if __name__ == "__main__":

    rightPast = deque([])
    leftPast = deque([])
    f = extract.FeatureExtractor(sys.argv[1])
    f.setStartPoint()


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

            feature = f.addPoint(time.time(),imbgr,imdepth)

            if feature.shape:
                ax.scatter(feature[14],feature[15],feature[16],c='r')
                ax.scatter(feature[17],feature[18],feature[19],c='g')
            plt.draw()


            cv2.imshow('Filtered', imgray2)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break
