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


    #plt.ion()
    #fig = plt.figure()
    #plt.xlim([0,640])
    #plt.ylim([0,255])


    while 1:
        try:
            #cv.ShowImage('Depth', get_depth())
            imbgr = np.array(extract.get_video())
            imdepth = extract.get_depth()

            greenmoments,greenhull = f.getCentralMoments(imbgr,'right')
            redmoments,redhull = f.getCentralMoments(imbgr,'left')

            cv2.drawContours(imbgr,[greenhull],-1,(0,255,0),2)
            cv2.drawContours(imbgr,[redhull],-1,(0,0,255),2)

            feature = np.nan_to_num(f.addPoint(time.time(),imbgr,imdepth))
            print feature

            if feature.shape:
                cv2.circle(imbgr,(int(feature[14]),int(feature[15])),3,(0,0,255),4)
                cv2.circle(imbgr,(int(feature[16]),int(feature[17])),3,(0,255,0),4)
                #plt.scatter(feature[14],480 - feature[15],c='r')
                #plt.scatter(feature[16],480 - feature[17],c='g')
            #plt.draw()


            cv2.imshow('Demo', imbgr)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break
