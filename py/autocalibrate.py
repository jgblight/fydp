import numpy as np 
import featureExtraction as fe
from collections import deque
import freenect
import os
import sys
import cv2
import cv
import csv

if __name__ == "__main__":
	  
	gfingerTrackList = deque([])
	rfingerTrackList = deque([])
	greencalibrated = False	
	redcalibrated = False

	gLow = [50, 112, 94]
	gHigh = [200, 118, 107]
	bLow = [50, 90, 165]
	bHigh = [200, 101, 174]
	rLow = [50, 168, 109]
	rHigh = [200, 214, 130]

while 1:
	try:
		green = fe.colourFilter(tuple(gLow),tuple(gHigh))
		blue = fe.colourFilter(tuple(bLow),tuple(bHigh))
		red = fe.colourFilter(tuple(rLow),tuple(rHigh))

		imbgr = fe.get_video()
        	imdepth = fe.get_depth()

		greenFingers = green.getColourContours(imbgr)
		redFingers = red.getColourContours(imbgr)

		imgray = cv2.cvtColor(imbgr,cv.CV_BGR2GRAY)
        	imgray2 = cv2.cvtColor(imgray,cv.CV_GRAY2BGR)

		cv2.drawContours(imgray2,greenFingers,-1,(255,0,0),2)
		cv2.drawContours(imgray2,redFingers,-1,(255,0,0),2)
	
		cv2.imshow('Filtered', imgray2)

		if greencalibrated == False and len(greenFingers) != 0:
			if len(gfingerTrackList) < 5:
				gfingerTrackList.append(len(greenFingers))
			else:
				_ = gfingerTrackList.popleft()
				gfingerTrackList.append(len(greenFingers))

			if gfingerTrackList.count(5) != len(gfingerTrackList):
		
		
				if len(greenFingers) != 5:
					gLow[1] = gLow[1] -1
					gLow[2] = gLow[2] - 1
					gHigh[1] = gHigh[1] + 1
					gHigh[2] = gHigh[2] + 1

			elif gfingerTrackList.count(5) == 5:
				greencalibrated = True

		if redcalibrated == False and len(redFingers) != 0:
			if len(rfingerTrackList) < 5:
				rfingerTrackList.append(len(redFingers))
			else:
				_ = rfingerTrackList.popleft()
				rfingerTrackList.append(len(redFingers))

			if rfingerTrackList.count(5) != len(rfingerTrackList):
		
		
				if len(greenFingers) != 5:
					rLow[1] = rLow[1] -1
					rLow[2] = rLow[2] - 1
					rHigh[1] = rHigh[1] + 1
					rHigh[2] = rHigh[2] + 1

			elif rfingerTrackList.count(5) == 5:
				redcalibrated = True
		
				

		print "green: " + str(len(greenFingers)) + " red: " + str(len(redFingers))

	except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 27:
            break

#pseudocode:
#- run initial filter
#- find blobs
#- check number of blogs, centroids, hull?
#- loop until 5 fingers found on each hand

