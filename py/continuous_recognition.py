import numpy as np
import pickle
import sys
import time
import cv2
import cv

from HMM_learning import ContinuousSignModel
import featureExtraction as fe


def main():
    modelfile = open(sys.argv[2])

    model = pickle.load(modelfile)
    f = fe.FeatureExtractor(sys.argv[1])
    f.setStartPoint()

    detected = False
    labels = model.labels

    while 1:
        try:
            imbgr = np.array(fe.get_video())
            imdepth = np.array(fe.get_depth())

            f.addPoint(time.time(),imbgr,imdepth)

            obs = np.nan_to_num(f.getFeatures())

            prediction = model.detect(obs)
            if prediction is not None:
                cv2.putText(imbgr,labels[prediction],(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),5)
                print labels[prediction]

            cv2.imshow("Demo",imbgr)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 32:
            break

if __name__ == '__main__':
    main()
