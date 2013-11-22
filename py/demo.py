import numpy as np
import cv
import cv2
import os
import sys
import featureExtraction as fe
from sklearn import svm
from sklearn import cross_validation, grid_search
import pprint
import csv
import pickle



if __name__ == "__main__":

    modelfile = open(sys.argv[2])
    model = pickle.load(modelfile)
    labels = pickle.load(modelfile)

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

            print labels[imbgrclass[0]]

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break