import os
import glob,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# make modules accessible for the script

import image_processing.vehicle_detection.compare_detection_groundtruth as compare


def plot_recognitionrate_distance():
    date = '2011_09_26'
    drive=56
    thresholds=['200']

    startFrame = 0
    maxFrame = 100
    alpha=50
    stepsize=10
    maxrange=80
    fig = plt.figure()
    distances = np.linspace(0, maxrange, (maxrange/stepsize)+1,dtype=int)
    values_per_threshold=[]
    for threshold in thresholds:
        datapath_left = '0056_03_0-100_t'+threshold
        datapath_right = '0056_02_0-100_t'+threshold
        matcher=compare.GroundtruthComparison()
        matcher.runComparison(date,drive,datapath_left,datapath_right,startFrame,maxFrame,alpha,stepsize)
        values=[]
        for distance in distances:
            if not distance in matcher.error_rate_per_distance:
                values.append(-1)
                continue
            values.append(matcher.error_rate_per_distance[distance])
        values_per_threshold.append(values)

    for i in range(len(thresholds)):
        print len(distances)
        print len(values_per_threshold[i])
        plt.plot(distances,values_per_threshold[i],label=thresholds[i])
    plt.axis([0,maxrange,0,1])
    plt.show()





if __name__ == "__main__":
    plot_recognitionrate_distance()