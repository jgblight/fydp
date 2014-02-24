import numpy as np 
import freenect
import os
import sys
import cv2
import cv
import csv
from matplotlib import pyplot as plt
import featureExtraction as fe

class kmeans1:


    def __init__(self,image):
        self.img = image
        self.pt1 = None
        self.pt2 = None
        self.filter = None
        self.displayImage = image
        self.gotRoi = False

def trykmeans(recordpath):
	for filename in os.listdir(recordpath):
        	if os.path.splitext(filename)[1] == ".ppm":
            		ppmpath = os.path.join(recordpath,filename)
            		break

	img = cv2.imread(ppmpath)
	imycrcb = cv2.cvtColor(img,cv.CV_BGR2YCrCb)
	Z = imycrcb.reshape((-1,3))
 
	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 15
	ret,label,center = cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((imycrcb.shape))
	cv2.imshow('res2', res2)
	cv2.waitKey(0)

	h, w, d = img.shape
	dist = cv2.sqrt(ret/(w * h))
	dist = dist[0] + 5
	print "center: " + str(center)
	print "ret: " + str(ret)
	print "dist: " + str(dist)
 
	#cv2.imshow('res2',res2)
	#cv2.waitKey(0)

	for c in center:
		print "c + dist: " + str(c + dist)
		trycolour = fe.colourFilter((c - dist), (c + dist))	
		tryfingers = trycolour.getColourContours(img)
		#cv2.drawContours(img,tryfingers,-1,(0,255,0),2)
		#cv2.imshow('contours', img)
		print str(len(tryfingers))
		if len(tryfingers) >= 4 and len(tryfingers) <= 10 :
			
			for contour in tryfingers :

				if cv2.contourArea(contour) < 20:
					continue

				if cv2.contourArea(contour) > 10000:
					fingers = False
					break

				x,y,w,h = cv2.boundingRect(contour)
				ratio = (w - x)/(h - y)

				if ratio <= .75 and ratio >= 1.5 :
					fingers = False
					break
				else:
					fingers = True

			if fingers == True:
				cv2.drawContours(img,tryfingers,-1,(0,255,0),2)
				cv2.imshow('contours', img)
				cv2.waitKey(0)

		#cv2.destroyAllWindows()

		


	#cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
    trykmeans(sys.argv[1])
