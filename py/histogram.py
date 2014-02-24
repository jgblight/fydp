import numpy as np 
import featureExtraction as fe
from collections import deque
import freenect
import os
import sys
import cv2
import cv
import csv
from matplotlib import pyplot as plt

class Histogram:
    def __init__(self,image):
        self.img = image
        self.pt1 = None
        self.pt2 = None
        self.filter = None
        self.displayImage = image
        self.gotRoi = False


def histogram(recordpath):
    	for filename in os.listdir(recordpath):
        	if os.path.splitext(filename)[1] == ".ppm":
           		ppmpath = os.path.join(recordpath,filename)
            		break

    	imbgr = cv2.imread(ppmpath)
	imycrcb = cv2.cvtColor(imbgr,cv.CV_BGR2YCrCb)

   	h = np.zeros((300,256,3))
    	bins = np.arange(256).reshape(256,1)
    	color = [ (255,0,0),(0,255,0),(0,0,255) ] 

    	for ch, col in enumerate(color):
    		hist_item = cv2.calcHist([imycrcb],[ch],None,[256],[0,255])
		cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
   		hist=np.int32(np.around(hist_item))
    		pts = np.column_stack((bins,hist))
    		cv2.polylines(h,[pts],False,col)

    	h=np.flipud(h)
    	cv2.imshow('colorhist',h)	

	Z = imycrcb.reshape((-1,3))
 
	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 10
	ret,label,center = cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((imycrcb.shape))
	cv2.imshow('res2', res2)
    	
	
	plt.plot(res2[:,:,1], res2[:,:,2], 'ro')
	plt.axis([0, 255, 0, 255])
	plt.show()
	cv2.waitKey(0)


if __name__ == "__main__":
    histogram(sys.argv[1])
