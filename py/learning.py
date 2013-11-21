import numpy as np
import cv
import cv2
import os
import featureExtraction as fe
from sklearn import svm
from sklearn import cross_validation, grid_search
import pprint
import csv
import pickle

train_folder = "fakenect-storage/static_train"

param_grid = [
  {'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000, 10000], 'gamma': [0.01, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']}
 ]

def getFeatures(folder):
    features = []
    with open(os.path.join(folder,"calibration.csv")) as csvfile:
        reader = csv.reader(csvfile)
        low = [ float(x) for x in reader.next()]
        high = [ float(x) for x in reader.next()]

    green = fe.colourFilter(tuple(low),tuple(high))
    for f in os.listdir(folder):
        if os.path.splitext(f)[1] == ".ppm":
            imbgr = cv2.imread(os.path.join(folder,f))
            hull = green.getColourHull(imbgr)
            features.append(fe.getFeatureVector(hull))
    return features
   

def getLabelledSets(folder):
    X = []
    Y = []
    labels = []
    index = 0
    for label in os.listdir(folder):
        label_path = os.path.join(folder,label)
        if os.path.isdir(label_path):
            labels.append(label)
            for capture in os.listdir(label_path):
                capture_path = os.path.join(label_path,capture)
                if os.path.isdir(capture_path):
                    features = getFeatures(capture_path)
                    X += features
                    Y += [index] * len(features)

            index += 1

    return np.array(X),np.array(Y),labels


if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)
    X,Y,labels = getLabelledSets(train_folder)
    trainX, testX, trainY, testY = cross_validation.train_test_split(X,Y,test_size=0.4,random_state=0)
    
    svclf = svm.SVC()

    gridclf = grid_search.GridSearchCV(svclf,param_grid)
    gridclf.fit(trainX,trainY)

    pp.pprint(gridclf.grid_scores_)
    print ""
    pp.pprint(gridclf.best_estimator_)
    print ""
    pp.pprint(gridclf.best_score_)
    print ""

    print gridclf.score(testX,testY)

    # persist model
    pickler = open("model.pkl","wb")
    pickle.dump(gridclf,pickler)








