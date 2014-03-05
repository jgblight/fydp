import numpy as np
import os
import os.path
import sys
import re
import cv2
import time
import pickle
import csv
import random

from sklearn import hmm
import featureExtraction as extract
from sklearn.cross_validation import StratifiedKFold
from sklearn.cluster import KMeans


K = 30
folds = 5

rgb_pattern = re.compile("r-\d+\.\d+-\d+\.ppm")
depth_pattern = re.compile("d-\d+\.\d+-\d+\.pgm")
filename_pattern = re.compile("\w-(?P<epoch>\d+\.\d+)-\d+\.p\wm")

def getTimestamp(filename):
    match = filename_pattern.match(filename)
    if match:
        return float(match.group("epoch"))


class FakenectReader:
    def __init__(self, folder):
        self.folder = folder
        all_files = os.listdir(folder)

        rgb_match = [rgb_pattern.match(x) for x in all_files]
        self.rgb_stack = sorted([m.group(0) for m in rgb_match if m],reverse=True)

        depth_match = [depth_pattern.match(x) for x in all_files]
        self.depth_stack = sorted([m.group(0) for m in depth_match if m],reverse=True)


    def __iter__(self):
        return self

    def next(self):
        if not len(self.rgb_stack) or not len(self.depth_stack):
            raise StopIteration
        else:
            rgb_file = self.rgb_stack.pop()
            rgb_time = getTimestamp(rgb_file)
            depth_file = self.depth_stack.pop()
            while len(self.depth_stack) and getTimestamp(self.depth_stack[-1]) <= rgb_time:
                depth_file = self.depth_stack.pop()
            
            imbgr = cv2.imread(os.path.join(self.folder,rgb_file))
            imdepth = cv2.imread(os.path.join(self.folder,depth_file))[:,:,0]
            return getTimestamp(rgb_file),imbgr,imdepth

class Clusters:
    def __init__(self, K):
        self.K = K
        self.estimator = KMeans(init="k-means++",n_clusters=self.K)
        self.n_features = 0

    #not sure if there's a way to vectorize this more
    def train(self,dataset):
        self.n_features = dataset[0].shape[1]
        all_samples = np.array([])
        for sample in dataset:
            if all_samples.size:
                all_samples = np.vstack([all_samples,sample]) 
            else:
                all_samples = sample
        self.estimator.fit(all_samples) 
        

    def classify(self,sample):
        clustered = self.estimator.predict(sample)
        return clustered

    def classify_set(self,samples):
        clustered = []
        for sample in samples:
            clustered.append(self.classify(sample))
        return clustered

class SignModel:
    def __init__(self,labels):
        self.labels = labels
        self.models = []
        self.clusters = None

    def get_labels(self):
        return self.labels

    def train(self,train_X,train_Y,N):
        self.models = []
        self.clusters = Clusters(30)
        self.clusters.train(train_X)

        for i,label in enumerate(labels):
            training_set = [x for x,y in zip(train_X,train_Y) if (y==i)]

            model = hmm.MultinomialHMM(N)
            discrete_obs = self.clusters.classify_set(training_set)
            model = model.fit(discrete_obs)

            self.models.append(model)


    def predict(self,obs):
        likelihoods = []
        for model in self.models:
            if model:
                likelihoods.append(model.score(self.clusters.classify(obs)))
            else:
                likelihoods.append(0)
        return np.nanargmax(likelihoods)

def getDataset(training_folder):
    #need to set up some sort of cross-validation
    labels = []
    dataset_X = []
    dataset_Y = []

    for label in os.listdir(training_folder):
        label_path = os.path.join(training_folder,label)
        if os.path.isdir(label_path):
            labels.append(label)
            label_index = len(labels) - 1
            print label
            for capture in os.listdir(label_path):
                capture_path = os.path.join(label_path,capture)
                if os.path.isdir(capture_path):
                    f = extract.FeatureExtractor(os.path.join(capture_path,"calibration.csv"))
                    f.setStartPoint()
                    for timestamp,imbgr,imdepth in FakenectReader(capture_path):
                        f.addPoint(timestamp,imbgr,imdepth)

                    feature = f.getFeatures()
                    if feature.size:
                        dataset_X.append(np.nan_to_num(feature))
                        dataset_Y.append(label_index)

    return labels,dataset_X,dataset_Y

def evaluateModels(labels,dataset_X,dataset_Y,modelname,N):
    #jiggle hidden state parameter

    confusion = np.zeros([len(labels),len(labels)])

    skf = StratifiedKFold(dataset_Y, folds)
    all_samples = 0
    correct = 0

    for train,test in skf:
        train_X = [ dataset_X[i] for i in train ]
        train_Y = [ dataset_Y[i] for i in train ]
        test_X = [ dataset_X[i] for i in test ]
        test_Y = [ dataset_Y[i] for i in test ]

        model = SignModel(labels)
        model.train(train_X,train_Y,N)

        for x,y in zip(test_X,test_Y):
            prediction = model.predict(x)
            if prediction is not np.nan:
                confusion[prediction,y] += 1
            if prediction == y:
                correct += 1
            all_samples += 1
    
    print "Accuracy"
    print correct / float(all_samples)

    precision = []
    recall = []

    for i in range(len(labels)):
        precision.append(confusion[i,i] / float(sum(confusion[i,:])))
        recall.append(confusion[i,i]/ float(sum(confusion[:,i])))

    print "Precision: " + str(precision)
    print "Recall: " + str(recall)

    print confusion

    with open(modelname+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Accuracy',correct / float(all_samples)])
        writer.writerow(['Precision']+ precision)
        writer.writerow(['Recall']+recall)

        writer.writerow(['Confusion Matrix'])
        writer.writerow(labels)
        for row in confusion:
            writer.writerow(row)

    return correct / float(all_samples)

def createModel(labels,dataset_X,dataset_Y,modelname,N):
    model = SignModel(labels)
    model.train(train_X,train_Y,N)
    # persist model
    pickler = open(modelname+".pkl","wb")
    pickle.dump(model,pickler)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        labels,dataset_X,dataset_Y = getDataset(sys.argv[2])

        # persist dataset
        pickler = open("dataset.pkl","wb")
        pickle.dump(labels,pickler)
        pickle.dump(dataset_X,pickler)
        pickle.dump(dataset_Y,pickler)
    else:
        #retrieve dataset
        modelfile = open("dataset.pkl")
        labels = pickle.load(modelfile)
        dataset_X = pickle.load(modelfile)
        dataset_Y = pickle.load(modelfile)


    max_accuracy = 0
    best_N = 0
    for N in range(3,11):
        print "N = " + str(N)
        accuracy = evaluateModels(labels,dataset_X,dataset_Y,sys.argv[1],N)
        if accuracy > max_accuracy:
            best_N = N
            max_accuracy = accuracy

    createModel(labels,dataset_X,dataset_Y,sys.argv[1],best_N)


