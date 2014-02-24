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

def tryslidingwindow(recordpath):
	for filename in os.listdir(recordpath):
        	if os.path.splitext(filename)[1] == ".ppm":
            		ppmpath = os.path.join(recordpath,filename)
            		break

	imbgr = cv2.imread(ppmpath)
	

	gfingerTrackList = deque([])
	rfingerTrackList = deque([])
	greencalibrated = False	
	redcalibrated = False

	gLow = [50, 0, 0]
	gHigh = [200, 20, 20]
	bLow = [50, 0, 120]
	bHigh = [200, 20, 140]
	rLow = [50, 120, 0]
	rHigh = [200, 140, 20]

	green = fe.colourFilter(tuple(gLow),tuple(gHigh))
	blue = fe.colourFilter(tuple(bLow),tuple(bHigh))
	red = fe.colourFilter(tuple(rLow),tuple(rHigh))
	
	greenFingers = green.getColourContours(imbgr)
	blueHands = blue.getColourContours(imbgr)
	redFingers = red.getColourContours(imbgr)

	greenFound = False
	blueFound = False
	redFound = False

#Green
	for i in range (1, 3):
		for r in range (1, 40):			
			widen(gHigh)
			green = fe.colourFilter(tuple(gLow),tuple(gHigh))
			greenFingers = green.getColourContours(imbgr)
			if len(greenFingers) == 5:
				greenFound = True
				break
		if greenFound == True:
			break
		for r in range (1, 40):			
			narrow(gHigh)
			green = fe.colourFilter(tuple(gLow),tuple(gHigh))
			greenFingers = green.getColourContours(imbgr)
			if len(greenFingers) == 5:
				greenFound = True				
				break
		if greenFound == True:
			break
#Blue
	for i in range (1, 3):
		for r in range (1, 40):			
			widen(bHigh)
			blue = fe.colourFilter(tuple(bLow),tuple(bHigh))
			blueHands = blue.getColourContours(imbgr)
			if len(blueHands) == 2:
				blueFound = True
				break
		if blueFound == True:
			break
		for r in range (1, 40):			
			narrow(bLow)
			blue = fe.colourFilter(tuple(bLow),tuple(bHigh))
			blueHands = blue.getColourContours(imbgr)
			if len(blueHands) == 2:
				blueFound = True
				break
		if blueFound == True:
			break
#Red
	for i in range (1, 3):
		for r in range (1, 40):			
			widen(rHigh)
			red = fe.colourFilter(tuple(rLow),tuple(rHigh))
			redFingers = red.getColourContours(imbgr)
			if len(redFingers) == 5:
				redFound = True
				break
		if redFound == True:
			break
		for r in range (1, 40):			
			narrow(rHigh)
			red = fe.colourFilter(tuple(rLow),tuple(rHigh))
			greenFingers = red.getColourContours(imbgr)
			if len(redFingers) == 5:
				redFound = True				
				break
		if redFound == True:
			break

	print "# green: " + str(len(greenFingers))
	print "# blue: " + str(len(blueHands))
	print "# red: " + str(len(redFingers))
	
	cv2.drawContours(imbgr,greenFingers,-1,(0,255,0),2)
	cv2.drawContours(imbgr,blueHands,-1,(255,0,0),2)
	cv2.drawContours(imbgr,redFingers,-1,(0,0,255),2)

	cv2.imshow('results', imbgr)
	cv2.waitKey(0)

def widen(high):
	high[1] = high[1] + 1
	high[2] = high[2] + 1

def narrow(low):
	low[1] = low[1] + 1
	low[2] = low[2] + 1
	

if __name__ == "__main__":
    tryslidingwindow(sys.argv[1])
