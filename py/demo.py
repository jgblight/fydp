import numpy as np
import cv
import cv2
import os
import sys
import featureExtraction as fe
from sklearn import svm
from sklearn import cross_validation, grid_search
from collections import deque
import pprint
import csv
import pickle


def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]


if __name__ == "__main__":

    modelfile = open(sys.argv[2])

    model = pickle.load(modelfile)
    labels = pickle.load(modelfile)
    running = deque([])

    with open(sys.argv[1]) as csvfile:
        reader = csv.reader(csvfile)
        low = [ float(x) for x in reader.next()]
        high = [ float(x) for x in reader.next()]

    green = fe.colourFilter(tuple(low),tuple(high))

    while 1:
        try:
            imbgr = fe.get_video()
            hull = green.getColourHull(imbgr)
            if len(hull):
                features = fe.getFeatureVector(hull)

                imbgrclass = model.predict([features])
                imbgrclass = imbgrclass[0]
                
                if len(running) < 10:
                    sign = imbgrclass
                    running.append(sign)
                else:
                    _ = running.popleft()
                    running.append(imbgrclass)
                    sign = median(running)

                sign = int(round(sign))

                letter = labels[sign]
            else:
                letter = ""


            imgray = cv2.cvtColor(imbgr,cv.CV_BGR2GRAY)

            cv2.drawContours(imgray,[hull],-1,(255,0,0),2)
            cv2.putText(imgray,letter,(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)

            cv2.imshow("Demo",imbgr)

            cv2.imshow("Hull",imgray)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break
