import numpy as np
import os
import os.path
import sys
import cv2
import cv
import pickle
import csv

import featureExtraction as fe

def blue_contours(cnt):
    cnt_sorted = sorted(cnt,key=cv2.contourArea)
    if len(cnt_sorted) >= 2:
        largest1 = cnt_sorted[-1]
        largest2 = cnt_sorted[-2]
        l1_m = cv2.moments(largest1)
        l2_m = cv2.moments(largest2)
        try:
            l1_x = int(l1_m['m10']/l1_m['m00'])
            l2_x = int(l2_m['m10']/l2_m['m00'])
        except ZeroDivisionError:
            return np.zeros(14)
        if l1_x < l2_x:
            blue_left_m = [l1_m['nu20'],l1_m['nu11'],l1_m['nu02'],l1_m['nu30'],l1_m['nu21'],l1_m['nu12'],l1_m['nu03']]
            blue_right_m = [l2_m['nu20'],l2_m['nu11'],l2_m['nu02'],l2_m['nu30'],l2_m['nu21'],l2_m['nu12'],l2_m['nu03']]
        else:
            blue_left_m = [l2_m['nu20'],l2_m['nu11'],l2_m['nu02'],l2_m['nu30'],l2_m['nu21'],l2_m['nu12'],l2_m['nu03']]
            blue_right_m = [l1_m['nu20'],l1_m['nu11'],l1_m['nu02'],l1_m['nu30'],l1_m['nu21'],l1_m['nu12'],l1_m['nu03']]
        return np.append(blue_left_m,blue_right_m)
    return np.zeros(14)

def get_features():
    calibration_folder = sys.argv[2]
    green_moments = np.zeros(7)
    red_moments = np.zeros(7)
    blue_moments = np.zeros(14)
    calibration = {}


    count = 0
    for record in os.listdir(calibration_folder):
        recordpath = os.path.join(calibration_folder,record)
        calibration_file = os.path.join(recordpath,'calibration.csv')
        if os.path.isdir(recordpath) and os.path.exists(calibration_file):
            for filename in os.listdir(recordpath):
                if os.path.splitext(filename)[1] == ".ppm":
                    ppmpath = os.path.join(recordpath,filename)
                    break

            count += 1
            imbgr = cv2.imread(ppmpath)
            f = fe.FeatureExtractor(calibration_file)
            green_m,_ = f.getCentralMoments(imbgr,'right')
            green_moments += green_m
            red_m,_ = f.getCentralMoments(imbgr,'left')
            red_moments += red_m

            cnt = f.markers['glove'].getColourContours(imbgr)
            blue_m = blue_contours(cnt)

            blue_moments += blue_m

            print f.markers['right'].low
            if calibration.has_key('glow'):
                calibration['glow'] = np.vstack((calibration['glow'],f.markers['right'].low))
            else:
                calibration['glow'] = f.markers['right'].low

            if calibration.has_key('ghigh'):
                calibration['ghigh'] = np.vstack((calibration['ghigh'],f.markers['right'].high))
            else:
                calibration['ghigh'] = f.markers['right'].high

            if calibration.has_key('blow'):
                calibration['blow'] = np.vstack((calibration['blow'],f.markers['glove'].low))
            else:
                calibration['blow'] = f.markers['glove'].low

            if calibration.has_key('bhigh'):
                calibration['bhigh'] = np.vstack((calibration['bhigh'],f.markers['glove'].high))
            else:
                calibration['bhigh'] = f.markers['glove'].high

            if calibration.has_key('rlow'):
                calibration['rlow'] = np.vstack((calibration['rlow'],f.markers['left'].low))
            else:
                calibration['rlow'] = f.markers['left'].low

            if calibration.has_key('rhigh'):
                calibration['rhigh'] = np.vstack((calibration['rhigh'],f.markers['left'].high))
            else:
                calibration['rhigh'] = f.markers['left'].high

    green_moments = green_moments / float(count)
    red_moments = red_moments / float(count)
    blue_moments = blue_moments / float(count)

    return green_moments,red_moments,blue_moments,calibration

def get_error(imbgr,low,high,ideal,blue=False):
    filt = fe.colourFilter(low,high)
    if blue:
        cnt = filt.getColourContours(imbgr)
        feature = blue_contours(cnt)
    else:
        hull = filt.getColourHull(imbgr)
        if len(hull):
            m = cv2.moments(hull)
            feature = np.array([m['nu20'],m['nu11'],m['nu02'],m['nu30'],m['nu21'],m['nu12'],m['nu03']])
        else:
            feature = np.array([0,0,0,0,0,0,0])

    dist = np.sqrt(np.sum(np.square(ideal-feature)))
    return dist

def get_start(c,low_key,high_key):

    low = np.zeros(3)
    high = np.zeros(3)
    for i in range(3):
        history = c[low_key][:,i]
        low[i] = np.random.uniform(np.min(history),np.max(history))
        history = c[high_key][:,i]
        high[i] = np.random.uniform(np.max([np.min(history),low[i]]),np.max(history))
    return low,high

def optimize(imbgr,got,frame_count,low,high,c,low_key,high_key,threshold,blue=False):
    iterations = 10
    if not got:
        i = 0
        error = get_error(imbgr,low,high,g_m,blue)
        if frame_count == 0:
            restart = np.random.uniform()
            if restart*error > threshold*1.5:
                low,high = get_start(c,low_key,high_key)
            while i < iterations:
                d = 10 * (iterations-i)/float(iterations)
                d_low = np.random.choice([-1,0,1],3)*d
                d_high = np.random.choice([-1,0,1],3)*d

                new_error = get_error(imbgr,low+d_low,high,g_m,blue)
                if new_error < error:
                    error = new_error
                    low = low+d_low 
                    
                new_error = get_error(imbgr,low,high+d_high,g_m,blue)
                if new_error < error:
                    error = new_error
                    high = high+d_high  

                i += 1  

        print error
        if error < threshold:
            frame_count += 1
        elif frame_count > 0:
            frame_count = 0

        if frame_count > 10:
            got = True
    return got,frame_count,low,high

def autocalibrate(g_m,r_m,b_m,c):

    got_green = False
    got_blue = False
    got_red = False

    gcount = 0
    bcount = 0
    rcount = 0
    frame_count = 0

    glow,ghigh = get_start(c,'glow','ghigh')
    blow,bhigh = get_start(c,'blow','bhigh')
    rlow,rhigh = get_start(c,'rlow','rhigh')

    while 1:
        try:
            imbgr = np.array(fe.get_video())

            print "green"
            got_green,gcount,glow,ghigh = optimize(imbgr,got_green,gcount,glow,ghigh,c,"glow","ghigh",0.04)
            print "red"
            got_red,rcount,rlow,rhigh = optimize(imbgr,got_red,rcount,rlow,rhigh,c,"rlow","rhigh",0.04)
            print "blue"
            got_blue,bcount,blow,bhigh = optimize(imbgr,got_blue,bcount,blow,bhigh,c,"blow","bhigh",0.05)
            
            green = fe.colourFilter(glow,ghigh)   
            hull = green.getColourHull(imbgr)   
            if len(hull):
                cv2.drawContours(imbgr,[hull],-1,(0,255,0),2) 

            red = fe.colourFilter(rlow,rhigh)   
            hull = red.getColourHull(imbgr)   
            if len(hull):
                cv2.drawContours(imbgr,[hull],-1,(0,0,255),2) 

            blue = fe.colourFilter(blow,bhigh)   
            cnt = blue.getColourContours(imbgr)
            if len(cnt) >= 2:
                cnt_sorted = sorted(cnt,key=cv2.contourArea)
                cv2.drawContours(imbgr,[cnt_sorted[-1]],-1,(255,0,0),2)  
                cv2.drawContours(imbgr,[cnt_sorted[-2]],-1,(255,0,0),2)  

            cv2.imshow("Demo",imbgr)

            if got_green and got_red and got_blue:
                frame_count += 1

        except KeyboardInterrupt:
            break
        if cv.WaitKey(10) == 32:
            break
        if frame_count >= 30:
            break

    with open(sys.argv[1],'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(glow)
        writer.writerow(ghigh)
        writer.writerow(blow)
        writer.writerow(bhigh)
        writer.writerow(rlow)
        writer.writerow(rhigh)



if __name__ == '__main__':
    if len(sys.argv) > 2:
        g_m,r_m,b_m,c = get_features()
        pickler = open("calibration_features.pkl","wb")
        pickle.dump(g_m,pickler)
        pickle.dump(r_m,pickler)
        pickle.dump(b_m,pickler)
        pickle.dump(c,pickler)
    else:
        modelfile = open("calibration_features.pkl")
        g_m = pickle.load(modelfile)
        r_m = pickle.load(modelfile)
        b_m = pickle.load(modelfile)
        c = pickle.load(modelfile)
    autocalibrate(g_m,r_m,b_m,c)


    