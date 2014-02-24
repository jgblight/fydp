import numpy as np 
import freenect
import os
import sys
import cv2
import cv
import csv
from collections import deque
from matplotlib import pyplot as plt
import featureExtraction as fe


class refine:

    def __init__(self,image):
        self.img = image
        self.pt1 = None
        self.pt2 = None
        self.filter = None
        self.displayImage = image
        self.gotRoi = False

def tryrefine(recordpath):
	for filename in os.listdir(recordpath):
        	if os.path.splitext(filename)[1] == ".ppm":
            		ppmpath = os.path.join(recordpath,filename)
            		break

	imbgr = cv2.imread(ppmpath)
	#imycrcb = cv2.cvtColor(imbgr,cv.CV_BGR2YCrCb)
	#imycrcb[:,:,1] = cv2.equalizeHist(imycrcb[:,:,1])
	#imbgr = cv2.cvtColor(imycrcb,cv.CV_YCrCb2BGR)

	gfingerTrackList = deque([])
	rfingerTrackList = deque([])
	greencalibrated = False	
	redcalibrated = False

	gLow = [50, 0, 0]
	gHigh = [200, 125, 125]
	bLow = [50, 0, 126]
	bHigh = [200, 126, 255]
	rLow = [50, 126, 90]
	rHigh = [200, 255, 150]

	green = fe.colourFilter(tuple(gLow),tuple(gHigh))
	blue = fe.colourFilter(tuple(bLow),tuple(bHigh))
	red = fe.colourFilter(tuple(rLow),tuple(rHigh))

	
	greenFingers = green.getColourContours(imbgr)
	blueHands = blue.getColourContours(imbgr)
	redFingers = red.getColourContours(imbgr)
	i = 1
	j = 1
	k = 1
	

	while len(greenFingers) != 5 and i < 50:		
		gLow[1] = gLow[1] +1
		gLow[2] = gLow[2] + 1
		gHigh[1] = gHigh[1] - 1
		gHigh[2] = gHigh[2] - 1
		green = fe.colourFilter(tuple(gLow),tuple(gHigh))
		greenFingers = green.getColourContours(imbgr)
		i = i +1

	while len(blueHands) !=2 and j < 50:
		bLow[1] = bLow[1] + 1
		bLow[2] = bLow[2] + 1
		bHigh[1] = bHigh[1] - 1
		bHigh[2] = bHigh[2] - 1
		blue = fe.colourFilter(tuple(bLow),tuple(bHigh))
		blueHands = blue.getColourContours(imbgr)
		j = j + 1
	
	
	while len(redFingers) != 5 and k < 50:
		rLow[1] = rLow[1] + 1
		rLow[2] = rLow[2] + 1
		rHigh[1] = rHigh[1] - 1
		rHigh[2] = rHigh[2] - 1
		red = fe.colourFilter(tuple(rLow),tuple(rHigh))
		redFingers = red.getColourContours(imbgr)

		k = k + 1

	print "# green: " + str(len(greenFingers))
	print "# blue: " + str(len(blueHands))
	print "# red: " + str(len(redFingers))

	cv2.drawContours(imbgr,greenFingers,-1,(0,255,0),2)
	cv2.drawContours(imbgr,blueHands,-1,(255,0,0),2)
	cv2.drawContours(imbgr,redFingers,-1,(0,0,255),2)

	cv2.imshow('results', imbgr)
	cv2.waitKey(0)

if __name__ == "__main__":
    tryrefine(sys.argv[1])
