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
    ax.set_xlim([0,640])
    ax.set_ylim([0,255])
    ax.set_zlim([0,480])


    while 1:
        try:
            #cv.ShowImage('Depth', get_depth())
            imbgr = np.array(extract.get_video())
            imdepth = extract.get_depth()

            greenmoments,greenhull = f.getCentralMoments(imbgr,'right')
            redmoments,redhull = f.getCentralMoments(imbgr,'left')

            cv2.drawContours(imbgr,[greenhull],-1,(0,255,0),2)
            cv2.drawContours(imbgr,[redhull],-1,(0,0,255),2)

            feature = f.addPoint(time.time(),imbgr,imdepth)
            print feature

            if feature.shape:
                ax.scatter(feature[14],255 - feature[16],480 - feature[15],c='r')
                ax.scatter(feature[17],255 - feature[19],480 - feature[18],c='g')
            plt.draw()


            cv2.imshow('Demo', imbgr)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break
