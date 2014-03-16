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
    cv2.resizeWindow("Demo", 1000, 900)
    folderpath = os.path.dirname(__file__)  
    modelfile = open(sys.argv[2])
    models = pickle.load(modelfile)
    labels = models.labels

    f = fe.FeatureExtractor(sys.argv[1])

    for i in range(3):
        try:

            f.setStartPoint()
            detectedSign = 0

            if i == 0:
                sign1 = 0
                sign2 = 8
                question = "A BIG pet dog or a SMALL one?"
                sign1Location = os.path.join(folderpath, "husky.png")
                sign2Location = os.path.join(folderpath, "chihuahua.png")
                showText1 = "A BIG dog!"
                showText2 = "A SMALL dog!"
            elif i == 1:
                sign1 = 3
                sign2 = 6
                question = "Stay at your HOUSE or go to a MOVIE?"
                sign1Location = os.path.join(folderpath, "lonely.jpg")
                sign2Location = os.path.join(folderpath, "popcorn.png")
                showText1 = "Booooring"
                showText2 = "Can I come?"

            elif i == 2:
                sign1 = 5
                sign2 = 1
                question = "Stuck on a desert island with your MOTHER or your CAT?"
                sign1Location = os.path.join(folderpath, "husky.png")
                sign2Location = os.path.join(folderpath, "grumpycat.jpg")             
                showText1 = "Your MOTHER??"
                showText2 = "Your CAT??"

            while detectedSign < 50:
                try:
                    
                    imbgr = np.array(fe.get_video())
                    img = np.copy(imbgr)

                    if not detectedSign:

                        imdepth = np.array(fe.get_depth())
                        cv2.putText(imbgr,question,(5,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)
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
                        cv2.putText(showimage, showtext ,(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)
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
