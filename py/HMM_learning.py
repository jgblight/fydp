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
from multiprocessing import Process,Pipe
from itertools import izip

from sklearn import hmm
import featureExtraction as extract
from sklearn.cross_validation import StratifiedKFold
from sklearn.cluster import KMeans

import hmmpy

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

class DiscreteSignModel:
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

            model = hmmpy.HMM(N,V=range(30))
            discrete_obs = self.clusters.classify_set(training_set)
            model = hmmpy.baum_welch(model,discrete_obs)

            self.models.append(model)


    def predict(self,obs):
        likelihoods = []
        for model in self.models:
            if model:
                likelihoods.append(hmmpy.forward(model,self.clusters.classify(obs))[0])
            else:
                likelihoods.append(0)
        return np.nanargmax(likelihoods)

class ContinuousSignModel:
    def __init__(self,labels):
        self.labels = labels
        self.models = []
        self.threshold = None

    def get_labels(self):
        return self.labels

    def create_threshold_model(self):
        N = []
        for m in self.models:
            N.append(m.n_components)
            n_mix = m.n_mix
        states = np.sum(N)
        A_t = np.zeros((states,states))
        gmms = []
        i_t = 0
        for m in self.models:
            gmms = gmms + m.gmms_
            for i_m in range(m.n_components):
                a = np.exp(m._log_transmat[i_m,i_m])
                A_t[i_t,:] = (1-a)/float(states-1)
                A_t[i_t,i_t] = a
                i_t += 1
        self.threshold = hmm.GMMHMM(states,n_mix,transmat=A_t,gmms=gmms)

    def train(self,train_X,train_Y,N):
        self.models = []

        for i,label in enumerate(labels):
            training_set = [x for x,y in zip(train_X,train_Y) if (y==i)]

            #A = np.zeros((N[i],N[i]))
            #for j in range(N[i]):
            #    A[j,j:j+4] = 1.0/(np.min((4,N[i]-j)))
            model = hmm.GMMHMM(N[i],N[-1],n_iter=20)
            model.fit(training_set)
            self.models.append(model)
        self.create_threshold_model()

    def predict(self,obs):
        likelihoods = []
        for model in self.models:
            likelihoods.append(model.score(obs))
        score = np.nanmax(likelihoods)
        prediction =  np.nanargmax(likelihoods)
        return prediction,score

    def get_score(self, obs, index):
        return self.models[index].score(obs)

    def get_threshold(self,obs):
        return self.threshold.score(obs)

    def detect(self,obs,index=None):
        cutoff = 0
        zero_counter = 0
        for i in range(obs.shape[0]):            
            if np.all(obs[i,:]==0):
                zero_counter += 1
            else:
                zero_counter = 0
            if zero_counter > 10:
                cutoff = i + 1

        obs = obs[cutoff:,:]

        if obs.shape[0] > 100:
            obs = obs[-100:,:]

        if obs.shape[0] >= 20:
            for i in range(0,obs.shape[0]/10):
                obs_short = obs[i*10:,:]
                if index is None:
                    prediction,score = self.predict(obs)
                else:
                    score = self.get_score(obs, index)
                    prediction = True
                threshold = self.get_threshold(obs_short)
                if score > threshold:
                    return prediction
            if index is None:
                return None
            return False

def getDataset(training_folder):
    #need to set up some sort of cross-validation
    labels = []
    dataset_X = []
    dataset_Y = []

    for label in os.listdir(training_folder):
        label_path = os.path.join(training_folder,label)
        if os.path.isdir(label_path) and not label == "GARBAGE":
            labels.append(label)
            label_index = len(labels) - 1
            print label
            for capture in os.listdir(label_path):
                capture_path = os.path.join(label_path,capture)
                if os.path.isdir(capture_path) and not capture in ['golf1','golf2']:
                    f = extract.FeatureExtractor(os.path.join(capture_path,"calibration.csv"))
                    f.setStartPoint()
                    for timestamp,imbgr,imdepth in FakenectReader(capture_path):
                        f.addPoint(timestamp,imbgr,imdepth)

                    feature = f.getFeatures()
                    if feature.size:
                        dataset_X.append(np.nan_to_num(feature[:-5,:]))
                        dataset_Y.append(label_index)

    return labels,dataset_X,dataset_Y

def fragment_sample(obs):
    frames = obs.shape[0]
    window = 20
    samples = int(np.floor(1.5*(frames/window)))
    fragments = [obs]
    if frames > window:
        for i in range(samples):
            start = np.random.randint(0,frames-window)
            fragment = obs[start:start+window,:]
            fragments.append(fragment)
    return fragments

def fragment(X,Y):
    fragmented_X = []
    fragmented_Y = []
    for x,y in zip(X,Y):
        fragments_X = fragment_sample(x)
        for f in fragments_X:
            fragmented_X.append(f)
            fragmented_Y.append(y)
    return fragmented_X,fragmented_Y

def evaluateModel(labels,dataset_X,dataset_Y,N):

    confusion = np.zeros([len(labels),len(labels)])

    skf = StratifiedKFold(dataset_Y, folds)
    all_samples = 0
    correct = 0

    for train,test in skf:
        train_X = [ dataset_X[i] for i in train ]
        train_Y = [ dataset_Y[i] for i in train ]
        test_X = [ dataset_X[i] for i in test ]
        test_Y = [ dataset_Y[i] for i in test ]

        #train_X,train_Y = fragment(train_X,train_Y)
        #test_X,test_Y = fragment(test_X,test_Y)

        model = ContinuousSignModel(labels)
        model.train(train_X,train_Y,N)

        for x,y in zip(test_X,test_Y):
            prediction,score = model.predict(x)
            if prediction is not np.nan:
                confusion[prediction,y] += 1
            if prediction == y:
                correct += 1
            all_samples += 1

    return (correct / float(all_samples)),confusion

def randomSearch(labels,dataset_X,dataset_Y):
    np.random.seed(3)

    i = 0
    n_parameters = len(labels)+1
    N = np.random.randint(3,10,n_parameters)
    accuracy = evaluateModel(labels,dataset_X,dataset_Y,N)
    print N
    print accuracy
    while i < 20:
        i += 1
        new_N = np.copy(N)
        new_N[np.random.randint(0,n_parameters)] = np.random.randint(3,10)
        new_accuracy,_ = evaluateModel(labels,dataset_X,dataset_Y,new_N)
        if new_accuracy > accuracy:
            N = new_N
            accuracy = new_accuracy
        print new_N
        print str(accuracy) + "   " + str(new_accuracy)

    return N

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]

def randomrandomSearch(labels,dataset_X,dataset_Y):

    def generateRandomModel(seed):
        print "start"
        np.random.seed(seed)
        N = np.random.randint(3,10,len(labels)+1)
        accuracy,_ = evaluateModel(labels,dataset_X,dataset_Y,N)
        print "finish"
        return (N,accuracy)

    results = parmap(generateRandomModel,range(20))

    print results
    accuracy = 0
    N = []
    for r in results:
        if r[1] > accuracy:
            N = r[0]
            accuracy = [1]
    return N


def createModel(labels,dataset_X,dataset_Y,modelname,N):
    model = ContinuousSignModel(labels)
    model.train(dataset_X,dataset_Y,N)
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


    best_N = randomrandomSearch(labels,dataset_X,dataset_Y)
    accuracy,confusion = evaluateModel(labels,dataset_X,dataset_Y,best_N)
    print 'Accuracy: ' + str(accuracy)
    print confusion

    createModel(labels,dataset_X,dataset_Y,sys.argv[1],best_N)


