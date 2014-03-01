import numpy as np
from HMM_learning import FakenectReader
import featureExtraction as extract
import os.path
import time
import cv2
import cv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

library = "/Users/jgblight/Dropbox/fakenect-storage/sign_library"

samples = [('BIG/sara3',"BIG"),
            ('CAT/sara2',"CAT"),
            ('FAVOURITE/jen5',"FAVOURITE"),
            ('HOUSE/sara1',"HOUSE"),
            ('MORE/jen4',"MORE"),
            ('MOTHER/sara1',"MOTHER"),
            ('MOVIE/sara3',"MOVIE"),
            ('RED/jen5',"RED"),
            ('SMALL/sara1',"SMALL"),
            ('SWEETHEART/sara3',"SWEETHEART")]


if __name__ == "__main__":

    for sample,label in samples:
        sample_path = os.path.join(library,sample)

        for timestamp,imbgr,imdepth in FakenectReader(sample_path):
            try:
                cv2.putText(imbgr,label,(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)

                cv2.imshow("Original",imbgr)
                time.sleep(0.000001)

            except KeyboardInterrupt:
                break
            if cv.WaitKey(10) == 27:
                break

