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

if __name__ == "__main__":

    modelfile = open(sys.argv[2])


    labels = pickle.load(modelfile)
    models = pickle.load(modelfile)
    f = fe.FeatureExtractor(sys.argv[1])


    for i in range(10):
        rand_sign = np.random.randint(0,len(labels))

        while 1:
            try:
                imbgr = np.zeros((480,640,3))
                cv2.putText(imbgr,labels[rand_sign],(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)

                cv2.imshow("Demo",imbgr)

            except KeyboardInterrupt:
                break
            if cv.WaitKey(10) == 32:
                break


        f.setStartPoint()

        while 1:
            try:
                imbgr = np.array(fe.get_video())
                imdepth = np.array(fe.get_depth())

                cv2.imshow("Demo",imbgr)
                f.addPoint(time.time(),imbgr,imdepth)

            except KeyboardInterrupt:
                break
            if cv.WaitKey(10) == 32:
                break

        obs = np.nan_to_num(f.getFeatures())

        scores = []
        for i in models:
            scores.append(i.score(obs))

        print scores
        score_min = np.amin(scores)
        score_max = np.amax(scores)

        scores = (scores - score_min) / (score_max - score_min)

        likelihood = scores[rand_sign]
        print likelihood

        if likelihood < 0.3 or likelihood == np.nan:
            message = "Try Again!"
        elif likelihood < 0.5:
            message = "Weak"
        elif likelihood < 0.8:
            message = "Good Job"
        else:
            message = "Excellent!"



        while 1:
            try:
                imbgr = np.array(fe.get_video())
                cv2.putText(imbgr,message,(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)

                cv2.imshow("Demo",imbgr)

            except KeyboardInterrupt:
                break
            if cv.WaitKey(10) == 32:
                break 
