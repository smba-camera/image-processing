import os
import glob,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# make modules accessible for the script

import image_processing.vehicle_detection.compare_detection_groundtruth as compare
''' uses vehicle detection to mark all vehicles on the kitti images '''

def runStereoEvaluation():
    date = '2011_09_26'
    drive=56
    datapath_left = '0056_03_0-10_t975'
    datapath_right = '0056_02_0-10_t975'
    startFrame = 0
    maxFrame = 10
    alpha=50
    matched_pairs=compare.runComparison(date,drive,datapath_left,datapath_right,startFrame,maxFrame,alpha)
    print matched_pairs





if __name__ == "__main__":
    runStereoEvaluation()