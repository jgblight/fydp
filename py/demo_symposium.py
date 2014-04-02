import numpy as np
import cv
import cv2
import os
import sys
import featureExtraction as fe
import autocalibrate as auto
import csv
import pickle
import time
import calendar
from HMM_learning import ContinuousSignModel

def featureWindow(imbgr,f,v):
    greenmoments,greenhull = f.getCentralMoments(imbgr,'right')
    redmoments,redhull = f.getCentralMoments(imbgr,'left')

    cv2.drawContours(imbgr,[greenhull],-1,(0,255,0),2)
    cv2.drawContours(imbgr,[redhull],-1,(0,0,255),2)

    if v.shape:
        v = np.nan_to_num(v)
        cv2.circle(imbgr,(int(v[14]),int(v[15])),3,(0,0,255),4)
        cv2.circle(imbgr,(int(v[16]),int(v[17])),3,(0,255,0),4)

    cv2.imshow("Features",imbgr)


if __name__ == "__main__":

    cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Demo", 950, 900)   
    modelfile = open(sys.argv[2])
    models = pickle.load(modelfile)
    labels = models.labels

    if sys.argv[1] == "auto":
        auto.AutoCalibrate("calibration.csv","/home/sara/Dropbox/fakenect-storage/calibration")
        f = fe.FeatureExtractor("calibration.csv")
    else:
        f = fe.FeatureExtractor(sys.argv[1])

   

    for i in range(10):
        try:
            rand_sign = np.random.randint(0,len(labels))
            f.setStartPoint()
            detectedSign = 0
            countDown = 0

            while detectedSign < 30:
                try:
                    
                    imbgr = np.array(fe.get_video())
                    img = np.copy(imbgr)

                    if i == 0 and countDown < 90:
                        imbgr = np.zeros((480,640,3))
                        if countDown < 30:
                            cv2.putText(imbgr, "3" ,(250,250),cv2.FONT_HERSHEY_COMPLEX,5,(255,255,255),5)
                        elif countDown < 60:
                            cv2.putText(imbgr, "2" ,(250,250),cv2.FONT_HERSHEY_COMPLEX,5,(255,255,255),5)
                        elif countDown < 90:
                            cv2.putText(imbgr, "1" ,(250,250),cv2.FONT_HERSHEY_COMPLEX,5,(255,255,255),5)
                        cv2.imshow("Demo",imbgr)
                        countDown += 1

                    else:
                        if not detectedSign:
                            imdepth = np.array(fe.get_depth())

                            if not detectedSign:
                                  cv2.putText(imbgr,labels[rand_sign],(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)

                            v = f.addPoint(time.time(),imbgr,imdepth)
                            

                            obs = np.nan_to_num(f.getFeatures())
                            detected = models.detect(obs,rand_sign)
                            if detected:
                                detectedSign = 1

                        #print feedback                        
                        if detectedSign and detectedSign < 30:
                                imbgr = np.zeros((480,640,3))
                                cv2.putText(imbgr, "Excellent!" ,(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)
                                cv2.imshow("Demo",imbgr)
                                detectedSign += 1


                        cv2.imshow("Demo",imbgr)
                        featureWindow(img,f,v)

                except KeyboardInterrupt:
                    break
                if cv.WaitKey(10) == 32:
                    break
                
        except KeyboardInterrupt:
                break


