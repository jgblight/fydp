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

            zero_indices = np.where(np.all(obs==0,1))[0]
            if len(zero_indices):
                cutoff = zero_indices[-1]
                if obs.shape[0] > cutoff+2:
                    obs = obs[cutoff+1:,:]
                else:
                    obs = np.array([])
            
            if obs.shape[0] > 100:
                obs = obs[-100:,:]

            if obs.shape[0] >= 20:
                prediction,score = model.predict(obs)
                threshold = model.get_threshold(obs)

                #print str(score) + "  " + str(threshold)

                if score > threshold:
                    detected = True
                    print labels[prediction] + "   " + str(obs.shape[0])
                else:
                    if detected:
                        detected = False
                        f.setStartPoint()
                        print "reset"
                    else:
                        for i in range(1,obs.shape[0]/10):
                            obs_short = obs[i*10:,:]
                            prediction,score = model.predict(obs_short)
                            threshold = model.get_threshold(obs_short)
                            if score>threshold:
                                detected = True
                                #prediction = prediction_short
                                print labels[prediction] + "   " + str(obs.shape[0])

                if detected:
                    cv2.putText(imbgr,labels[prediction],(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),5)
                else:
                    cv2.putText(imbgr,labels[prediction],(5,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)


            cv2.imshow("Demo",imbgr)

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 32:
            break

if __name__ == '__main__':
    main()
