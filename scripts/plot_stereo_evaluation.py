import os
import glob,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

from image_processing.vehicle_detection.vehicle_detection_analyzer import VehicleDetectionAnalyzer
''' uses vehicle detection to mark all vehicles on the kitti images '''

def runStereoEvaluation():
    date = '2011_09_26'
    drive='0056'
    datapath_left = '0056_03_0-100_t200'
    datapath_right = '0056_02_0-100_t200'
    startFrame = 0
    maxFrame = 10
    alpha=50
    comparer = VehicleDetectionAnalyzer()
    comparer.runComparison(date,drive,datapath_left,datapath_right,startFrame,maxFrame,alpha)
    print(comparer.detection_rate)
    print(comparer.num_false_positives)
    print(comparer.error_rate_per_distance)
    print(comparer.x_mean_deviation)
    print(comparer.y_mean_deviation)
    print(comparer.x_mean_deviation_per_distance)
    print(comparer.y_mean_deviation_per_distance)

if __name__ == "__main__":
    runStereoEvaluation()