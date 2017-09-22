import os
import glob,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# make modules accessible for the script

import image_processing.vehicle_detection.compare_detection_groundtruth as compare


def plot_xy_deviation_per_distance():
    date = '2011_09_26'
    drive = 56
    startFrame = 0
    maxFrame = 10
    alpha = 50
    stepsize = 10
    maxrange = 80
    fig = plt.figure()
    distances = np.linspace(0, maxrange, (maxrange / stepsize) + 1, dtype=int)
    datapath_left = '0056_03_0-100_t200'
    datapath_right = '0056_02_0-100_t200'
    matcher = compare.GroundtruthComparison()
    matcher.runComparison(date, drive, datapath_left, datapath_right, startFrame, maxFrame, alpha, stepsize)
    xValues = []
    yValues = []
    for distance in distances:
        if not distance in matcher.x_mean_deviation_per_distance:
            xValues.append(-1)
        else:
            xValues.append(matcher.x_mean_deviation_per_distance[distance])
        if not distance in matcher.y_mean_deviation_per_distance:
            yValues.append(-1)
        else:
            yValues.append(matcher.y_mean_deviation_per_distance[distance])
    print xValues,yValues
    plt.plot(distances, xValues, label='X deviation')
    plt.plot(distances, yValues, label='Y deviation')
    plt.axis([0, maxrange, 0, 1])
    plt.show()





if __name__ == "__main__":
    plot_xy_deviation_per_distance()