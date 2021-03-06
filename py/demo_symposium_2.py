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

    #cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Demo", 1000, 900)
    folderpath = os.path.dirname(__file__)  
    modelfile = open(sys.argv[2])
    models = pickle.load(modelfile)
    labels = models.labels

    cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Demo", 950, 900)

    if sys.argv[1] == "auto":
        auto.AutoCalibrate("calibration.csv","/Users/jgblight/Dropbox/fakenect-storage/calibration")
        f = fe.FeatureExtractor("calibration.csv")
    else:
        f = fe.FeatureExtractor(sys.argv[1])

    countDown = 0

    for i in range(3):
        try:

            f.setStartPoint()
            detectedSign = 0

            if i == 0:
                sign1 = 0
                sign2 = 8
                question = "A BIG pet dog or a SMALL one?"
                question_l2 = ""
                sign1Location = os.path.join(folderpath, "husky.png")
                sign2Location = os.path.join(folderpath, "chihuahua.png")
                showText1 = "A BIG dog!"
                showText2 = "A SMALL dog!"
            elif i == 1:
                sign1 = 3
                sign2 = 6
                question = "Stay at your HOUSE or go to a MOVIE?"
                question_l2 = ""
                sign1Location = os.path.join(folderpath, "lonely.jpg")
                sign2Location = os.path.join(folderpath, "popcorn.jpg")
                showText1 = "Booooring"
                showText2 = "Can I come?"

            elif i == 2:
                sign1 = 5
                sign2 = 1
                question = "Stuck on a desert island with "
                question_l2 = "your MOTHER or your CAT?"
                sign1Location = os.path.join(folderpath, "husky.png")
                sign2Location = os.path.join(folderpath, "grumpycat.jpg")             
                showText1 = "Your MOTHER??"
                showText2 = "Your CAT??"

            while detectedSign < 50:
                try:
                    
                    #cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
                    #cv2.resizeWindow("Demo", 950, 900)
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
                            cv2.putText(imbgr,question,(5,50),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,0),2)
                            cv2.putText(imbgr,question_l2,(5,100),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,0),2)
                            v = f.addPoint(time.time(),imbgr,imdepth)
                          
                            obs = np.nan_to_num(f.getFeatures())

                            #sign1Detected = models.detect(obs, sign1)
                            #sign2Detected = models.detect(obs, sign2)
                            sign = models.detect(obs,[sign1,sign2])

                            cv2.imshow("Demo",imbgr)
     
                            if sign == sign1:
                                showimage = cv2.imread(sign1Location,0)
                                showtext = showText1
                                detectedSign = 1

                            if sign == sign2:
                                showimage = cv2.imread(sign2Location,0)
                                showtext = showText2
                                detectedSign = 1

                        if detectedSign and detectedSign < 50:
                            cv2.putText(showimage, showtext ,(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),4)
                            cv2.imshow("Demo",showimage)
                            detectedSign += 1


                        #cv2.imshow("Demo",imbgr)
                        featureWindow(img,f,v)

                except KeyboardInterrupt:
                    break
                if cv.WaitKey(10) == 32:
                    break
                
        except KeyboardInterrupt:
                break
