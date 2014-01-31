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
import cPickle as pickle

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
        low = [ float(x) for x in reader.next()]
        high = [ float(x) for x in reader.next()]
        blue = fe.colourFilter(tuple(low),tuple(high))

    for f in os.listdir(folder):
        if os.path.splitext(f)[1] == ".ppm":
            imbgr = cv2.imread(os.path.join(folder,f))
            hull = green.getColourHull(imbgr)
<<<<<<< HEAD
            features.append(fe.getFeatureVector(hull,['zernike']))
=======
            features.append(fe.getFeatureVector(hull,['central']))
            hull = blue.getColourHull(imbgr)
            features.append(fe.getFeatureVector(hull,['central']))
>>>>>>> blue
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

    train_folder = sys.argv[1]
    modelname = sys.argv[2]

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

    predictedY = gridclf.predict(testX)

    confusion = np.zeros([len(labels),len(labels)])
    for predicted,actual in zip(predictedY,testY):
        confusion[predicted,actual] += 1

    counts = []
    precision = []
    recall = []

    for i in range(len(labels)):
        precision.append(confusion[i,i] / float(sum(confusion[i,:])))
        recall.append(confusion[i,i]/ float(sum(confusion[:,i])))
        counts.append(sum(confusion[:,i]))


    with open(modelname+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cross-Validated Accuracy',gridclf.best_score_])
        writer.writerow(['Precision']+ precision)
        writer.writerow(['Recall']+recall)

        writer.writerow(['Confusion Matrix'])
        writer.writerow(labels)
        for row in confusion:
            writer.writerow(row)

    # persist model
    pickler = open(modelname+".pkl","wb")
    pickle.dump(gridclf.best_estimator_,pickler)
    pickle.dump(labels,pickler)








