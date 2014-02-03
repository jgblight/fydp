import numpy as np
import os
import os.path
import sys
import re
import cv2
import time
import pickle

from sklearn import hmm
import featureExtraction as extract

N = 6
test_percentage = 0.3

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

def getDataset(training_folder):
    #need to set up some sort of cross-validation
    labels = []
    training_data = {}
    training_data_paths = {}

    for label in os.listdir(training_folder):
        label_path = os.path.join(training_folder,label)
        if os.path.isdir(label_path):
            labels.append(label)
            training_data[label] = []
            training_data_paths[label] = []
            print label
            for capture in os.listdir(label_path):
                capture_path = os.path.join(label_path,capture)
                if os.path.isdir(capture_path):
                    f = extract.FeatureExtractor(os.path.join(capture_path,"calibration.csv"))
                    f.setStartPoint()
                    for timestamp,imbgr,imdepth in FakenectReader(capture_path):
                        f.addPoint(timestamp,imbgr,imdepth)

                    training_data[label].append(np.nan_to_num(f.getFeatures()))
                    training_data_paths[label].append(capture_path)

    return labels,training_data,training_data_paths

def trainModels(labels,training_data,training_data_paths,modelname):
    models = {}
    test_data = {}
    test_data_paths = {}
    #jiggle hidden state parameter
    for label in labels:
        training_set = []
        test_data[label] = []
        test_data_paths[label] = []
        for capture,capture_path in zip(training_data[label],training_data_paths[label]):
            if np.random.uniform() > test_percentage:
                training_set.append(capture)
            else:
                test_data[label].append(capture)
                test_data_paths[label].append(capture_path) 

        created_model = False
        n = N
        while not created_model and n > 0:
            try:
                print n
                models[label] = hmm.GMMHMM(n,3) #not sure how to make this a left-right HMM
                models[label].fit(training_set)
                created_model = True
            except ValueError:
                n-=1
        if not created_model:
            print "MODEL FAILED"

    print "testing"
    all_samples = 0
    correct = 0

    confusion = np.zeros([len(labels),len(labels)])

    for i,label in enumerate(labels):
        for sample in test_data[label]:
            maxlikelihood = -100000
            maxlabel = ""
            for j,model_label in enumerate(labels):
                if models.has_key(model_label):
                    likelihood = models[model_label].score(sample)
                    print likelihood
                    if likelihood > maxlikelihood:
                        maxlikelihood = likelihood
                        maxlabel = j
            all_samples += 1
            if maxlabel == i:
                correct += 1
            confusion[maxlabel,i] += 1
    print " "
    print correct / float(all_samples)

    precision = []
    recall = []

    for i in range(len(labels)):
        precision.append(confusion[i,i] / float(sum(confusion[i,:])))
        recall.append(confusion[i,i]/ float(sum(confusion[:,i])))

    print "Precision: " + str(precision)
    print "Recall: " + str(recall)

    # persist model
    pickler = open(modelname+".pkl","wb")
    pickle.dump(labels,pickler)
    pickle.dump(models,pickler)
    pickle.dump(test_data,pickler)
    pickle.dump(test_data_paths,pickler)

if __name__ == "__main__":
    #labels,training_data,training_data_paths = getDataset(sys.argv[1])

    # persist dataset
    #pickler = open("dataset.pkl","wb")
    #pickle.dump(labels,pickler)
    #pickle.dump(training_data,pickler)
    #pickle.dump(training_data_paths,pickler)

    #retrieve dataset
    modelfile = open("dataset.pkl")
    labels = pickle.load(modelfile)
    training_data = pickle.load(modelfile)
    training_data_paths = pickle.load(modelfile)

    trainModels(labels,training_data,training_data_paths,sys.argv[2])
