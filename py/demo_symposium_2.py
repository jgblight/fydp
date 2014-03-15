import numpy as np
import cv
import cv2
import os
import sys
import featureExtraction as fe
import csv
import pickle
import time
import calendar
from HMM_learning import ContinuousSignModel

if __name__ == "__main__":

    cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Demo", 1000, 900)   
    modelfile = open(sys.argv[2])
    models = pickle.load(modelfile)
    labels = models.labels

    f = fe.FeatureExtractor(sys.argv[1])

    for i in range(2):
        try:

            big = 0
            small = 8
            f.setStartPoint()
            detectedSign = 0

            while detectedSign < 30:
                try:
                    
                    imbgr = np.array(fe.get_video())

                    if not detectedSign:

                        imdepth = np.array(fe.get_depth())

                        cv2.putText(imbgr,"A big pet dog or a small one?",(5,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)

                        v = f.addPoint(time.time(),imbgr,imdepth)
                      
                        obs = np.nan_to_num(f.getFeatures())
                        bigDetected = models.detect(obs, big)
                        smallDetected = models.detect(obs, small)
                        
                        if bigDetected:
                            showimage = cv2.imread("chihuahua.png",0)
                            showtext = "A big dog!"
                            detectedSign = 1

                        if smallDetected:
                            showimage = cv2.imread("chihuahua.png",0)
                            showtext = "A small dog!"
                            detectedSign = 1

                    if detectedSign and detectedSign < 30:
                        cv2.putText(showimage, showtext ,(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)
                        cv2.imshow("test",showimage)
                        detectedSign += 1


                    cv2.imshow("Demo",imbgr)

                except KeyboardInterrupt:
                    break
                if cv.WaitKey(10) == 32:
                    break
                
        except KeyboardInterrupt:
                break
