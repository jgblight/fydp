import numpy as np
import os
import os.path
import sys
import re
import cv2
import time

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

def trainModels(training_folder):
    #need to set up some sort of cross-validation
    success = False
    labels = []
    models = {}
    test_data = {}

    for label in os.listdir(training_folder):
        label_path = os.path.join(training_folder,label)
        if os.path.isdir(label_path):
            labels.append(label)
            test_data[label] = []
            print label
            training_data = []
            for capture in os.listdir(label_path):
                capture_path = os.path.join(label_path,capture)
                if os.path.isdir(capture_path):
                    f = extract.FeatureExtractor(os.path.join(capture_path,"calibration.csv"))
                    f.setStartPoint()
                    for timestamp,imbgr,imdepth in FakenectReader(capture_path):
                        f.addPoint(timestamp,imbgr,imdepth)

                    if np.random.uniform < test_percentage:
                        test_data[label].append(np.nan_to_num(f.getFeatures()))
                    else:
                        training_data.append(np.nan_to_num(f.getFeatures()))

            #jiggle hidden state parameter
            n = N
            while not success:
                try:
                    models[label] = hmm.GaussianHMM(n) #not sure how to make this a left-right HMM
                    models[label].fit(training_data)
                    success = True
                except ValueError:
                    n-=1

    print "testing"
    all_samples = 0
    correct = 0

    for label in labels:
        for sample in test_data[label]:
            maxlikelihood = 0
            maxlabel = ""
            for model_label in labels:
                likelihood = models[model_label].predict(sample)
                if likelihood > maxlikelihood:
                    maxlikelihood = likelihood
                    maxlabel = model_label
            all_samples += 1
            if maxlabel == model_label:
                correct += 1
    print correct / float(all_samples)



if __name__ == "__main__":
    trainModels(sys.argv[1])

