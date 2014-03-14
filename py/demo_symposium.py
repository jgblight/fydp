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
    for i in range(10):
        try:
            rand_sign = np.random.randint(0,len(labels))
            f.setStartPoint()
            holdText = 0
            detectedSign = 0

            while detectedSign < 30:
                try:
                    
                    imbgr = np.array(fe.get_video())

                    if not detectedSign:
                        imdepth = np.array(fe.get_depth())

                        if holdText < 20:
                              cv2.putText(imbgr,labels[rand_sign],(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),5)
                              holdText += 1

                        v = f.addPoint(time.time(),imbgr,imdepth)
                        

                        obs = np.nan_to_num(f.getFeatures())

                        #To get rid of extra zeroes
                        zero_indices = np.where(np.all(obs==0,1))[0]
                        if len(zero_indices):
                            cutoff = zero_indices[-1]
                            if obs.shape[0] > cutoff+2:
                                obs = obs[cutoff+1:,:]
                            else:
                                obs = np.array([])

 
                        if obs.shape[0] > 100:
                            obs = obs[-100:,:]

                        if obs.shape[0] >= 20:
                            score = models.get_score(obs, rand_sign)
                            threshold = models.get_threshold(obs)
                            #print str(score) + "/" + str(threshold)
                            if score > threshold:
                                    detectedSign = 1
                            else:
                                for i in range(1,obs.shape[0]/10):
                                    obs_short = obs[i*10:,:]
                                    score = models.get_score(obs_short, rand_sign)
                                    threshold = models.get_threshold(obs_short)
                                    #print str(score) + "/" + str(threshold)
                                    if score>threshold:
                                        detectedSign = 1
                                        break


                    #print feedback                        
                    if detectedSign and detectedSign < 30:
                            imbgr = np.zeros((480,640,3))
                            cv2.putText(imbgr, "Excellent!" ,(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)
                            cv2.imshow("Demo",imbgr)
                            detectedSign += 1


                    cv2.imshow("Demo",imbgr)

                except KeyboardInterrupt:
                    break
                if cv.WaitKey(10) == 32:
                    break
                
        except KeyboardInterrupt:
                break


