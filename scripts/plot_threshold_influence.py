import os
import glob,sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# make modules accessible for the script

import image_processing.vehicle_detection.vehicle_detection_analyzer as compare

def plot_threshold_influence():
    date = '2011_09_26'
    drives = ['0022','0056']
    thresholds=['100','160','200','240']#['100','160','180','200','220']
    startFrame = 0
    maxFrame = 100
    alpha = 10
    falsePositives = [0] * len(thresholds)
    falseNegatives = [0] * len(thresholds)
    matcher = compare.VehicleDetectionAnalyzer()
    fig = plt.figure()
    for drive in drives:

        i=0
        for thresh in thresholds:
            datapath_left = drive + '_03_t'+thresh
            datapath_right = drive + '_02_t'+thresh
            matcher.runComparison(date, drive, datapath_left, datapath_right, alpha)
            falseNegatives[i]+=matcher.num_false_negatives
            print falseNegatives
            falsePositives[i]+=matcher.num_false_positives
           # print falseNegatives,falsePositives
            matcher.reset()
            i += 1
    aggregated_errors=[sum(x) for x in zip(*[falsePositives,falseNegatives])]
    thresholds=[int(x)/255.0 for x in thresholds ]
    plt.plot(thresholds, falseNegatives, 'bo-',label='false negatives',color='green')
    plt.plot(thresholds, falsePositives, 'bo-',label='false positives',color='blue')
    plt.plot(thresholds, aggregated_errors, 'bo-', label='false positives', color='red')

    plt.legend(loc='best')
    #plt.axis([np.min(thresholds)-10, np.max(thresholds)+10, max(np.max(falsePositives),np.max(falseNegatives))+1])
    #plt.ylabel('Average deviation from correct position per meter distance [m]')
    #plt.xlabel('Distance from detected cars [m]')
    #plt.title('Average deviation of calculated distance to real distance\n as measured on 3000 Kitti image pairs')
    #fig.savefig(os.path.join('data', 'plots', 'plot_xy_deviation_per_distance.png'))
    plt.show()
    #fig.savefig('data/plots/Stereo_Recognition_Rate_per_Distance.png')





if __name__ == "__main__":
    plot_threshold_influence()