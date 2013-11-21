import numpy as np
import cv
import cv2
import os
import featureExtraction as fe
from sklearn import svm
from sklearn import cross_validation, grid_search
import pprint

train_folder = "fakenect-storage/static_train"

param_grid = [
  {'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000, 10000], 'gamma': [0.01, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']}
 ]

def get_images(folder):
	images = {}
	for label in os.listdir(folder):
		label_path = os.path.join(folder,label)
		if os.path.isdir(label_path):
			images[label] = []
			for capture in os.listdir(label_path):
				capture_path = os.path.join(label_path,capture)
				if os.path.isdir(capture_path):
					for f in os.listdir(capture_path):
						if os.path.splitext(f)[1] == ".ppm":
							images[label].append(cv2.imread(os.path.join(capture_path,f)))
	return images

def getLabelledSets(images):
	X = []
	Y = []
	labels = []
	index = 0
	green = fe.colourFilter(cv.Scalar(50,115,90),cv.Scalar(150,135,110))
	for label in images.keys():
		labels.append(label)
		for im in images[label]:
			hull = green.getColourHull(im)
			X.append(fe.getFeatureVector(hull))
			Y.append(index)
		index += 1
	return np.array(X),np.array(Y),labels


if __name__ == "__main__":

	pp = pprint.PrettyPrinter(indent=4)
	images = get_images(train_folder)

	X,Y,labels = getLabelledSets(images)
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








