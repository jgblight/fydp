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
    last10labels = deque([])

    with open(sys.argv[1]) as csvfile:
        reader = csv.reader(csvfile)
        low = [ float(x) for x in reader.next()]
        high = [ float(x) for x in reader.next()]

    green = fe.colourFilter(tuple(low),tuple(high))

    while 1:
        try:
            #cv.ShowImage('Depth', get_depth())
            imbgr = fe.get_video()
            cv2.imshow("Demo",imbgr)
            hull = green.getColourHull(imbgr)
            features = fe.getFeatureVector(hull)

            imbgrclass = model.predict([features])
            imbgrclass = imbgrclass[0]
            
            if len(last10labels) < 10:
                sign = imbgrclass
                last10labels.append(sign)
            else:
                sixth = last10labels.popleft()
                last10labels.append(imbgrclass)
                sign = median(last10labels)

            sign = int(round(sign))

            print labels[sign]

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break