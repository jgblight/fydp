import numpy as np 
import featureExtraction as fe
from collections import deque
import freenect
import os
import sys
import cv2
import cv
import csv

delta = 2

class Sampler:

    def __init__(self,image):
        self.img = image
        self.pt1 = None
        self.pt2 = None
        self.filter = None
        self.displayImage = image
        self.gotRoi = False

    def applyFilter(self):
        imycrcb = cv2.cvtColor(self.img,cv.CV_BGR2YCrCb)
        lowx = min(self.pt1[0],self.pt2[0])
        highx = max(self.pt1[0],self.pt2[0])
        lowy = min(self.pt1[1],self.pt2[1])
        highy = max(self.pt1[1],self.pt2[1])
        y = []
        cr = []
        cb = []

        for i in range(lowx,highx+1):
            for j in range(lowy,highy+1):
                y.append(imycrcb[i,j,0])
                cr.append(imycrcb[i,j,1])
                cb.append(imycrcb[i,j,2])

        ymean = np.mean(y)
        crmean = np.mean(cr)
        cbmean = np.mean(cb)
        ystd = np.std(y)
        crstd = np.std(cr)
        cbstd = np.std(cb)

        print (ymean,crmean,cbmean)
        print (ystd,crstd,cbstd)

        self.filter = fe.colourFilter((50,crmean-crstd*delta,cbmean-cbstd*delta),(200,crmean+crstd*delta,cbmean+cbstd*delta))
        hull = self.filter.getColourHull(self.img)
        print self.filter.low
        print self.filter.high

        imgray = cv2.cvtColor(self.img,cv.CV_BGR2GRAY)
        cv2.drawContours(imgray,[hull],-1,(255,0,0),2)
        self.displayImage = imgray

def callback(sampler, pt1, pt2):

    def onmouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not sampler.gotRoi:
                sampler.pt1 = pt1
		sampler.pt2 = pt2
		sampler.applyFilter()
                sampler.gotRoi = True

    return onmouse

def getColourRange(writer,colour, pt1, pt2,regioncolour):
    
    windowname = "Select " + colour

    while 1: 
        try:
		imbgr = fe.get_video()   
        	imdepth = fe.get_depth()
		
		sampler = Sampler(imbgr)

		imgray = cv2.cvtColor(imbgr,cv.CV_BGR2GRAY)
		imgray2 = cv2.cvtColor(imgray,cv.CV_GRAY2BGR)
	
		cv2.rectangle(imgray2, pt1, pt2, regioncolour, 2)

		sampler = Sampler(imbgr)
    		
    		cv2.namedWindow(windowname,1)
    		cv2.setMouseCallback(windowname, callback(sampler, pt1, pt2))
		cv2.imshow(windowname, imgray2) 
		if cv.WaitKey(10) == 27:
           		break

	except KeyboardInterrupt:
            break
       
    cv2.destroywindow(windowname)
    writer.writerow(sampler.filter.low)
    writer.writerow(sampler.filter.high)

def calibrate(recordpath):

    with open(os.path.join(recordpath,'calibration.csv'),'w') as csvfile:
        writer = csv.writer(csvfile)
	print "Blue: "
        getColourRange(writer,"Blue", (200,300),(250,350), (255,0,0))
	print "Green: "
	getColourRange(writer,"Green", (200,300),(250,350), (0,255,0))


if __name__ == "__main__":
	calibrate(sys.argv[1])


