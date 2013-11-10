#!/usr/bin/env python
import freenect
import numpy as np
import cv
import cv2
import frame_convert
import time

global color

def get_depth():
    return frame_convert.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert.video_cv(freenect.sync_get_video()[0])


cv2.namedWindow("Original",1)

while 1:
    try:
        imbgr = get_video()
        cv2.imshow("Original",imbgr)
        time.sleep(0.000001)

    except KeyboardInterrupt:
        break
    if cv.WaitKey(10) == 27:
        break