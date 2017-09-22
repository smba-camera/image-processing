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
    drives = ['0001', '0005', '0009', '0015', '0019', '0022', '0028', '0056']
    startFrame = 0
    maxFrame = 100
    alpha = 10
    stepsize = 10
    maxrange = 80
    xValues = [0] * ((maxrange / stepsize) + 1)
    yValues = [0] * ((maxrange / stepsize) + 1)
    matcher = compare.GroundtruthComparison()
    fig = plt.figure()
    distances = np.linspace(1, maxrange, (maxrange / stepsize) + 1, dtype=int)
    for drive in drives:
        datapath_left = drive + '_03_t200'
        datapath_right = drive + '_02_t200'
        matcher.runComparison(date, drive, datapath_left, datapath_right, alpha, stepsize)
    i=0
    for distance in distances:
        if distance in matcher.x_mean_deviation_per_distance:
            xValues[i]=(matcher.x_mean_deviation_per_distance[distance])
        else:
            xValues[i] = None
        if distance in matcher.y_mean_deviation_per_distance:
            yValues[i]=(matcher.y_mean_deviation_per_distance[distance])
        else:
            yValues[i] = None
        i+=1

    distances = [x - 5 for x in distances]
    plt.plot(distances, xValues, 'bo', distances, xValues, 'k',label='x-direction')
    plt.plot(distances, yValues, 'bo', distances, yValues, 'k',label='y-direction')
    plt.axis([0, maxrange, 0, 3])
    plt.ylabel('Average deviation from correct position per meter distance [m]')
    plt.xlabel('Distance from detected cars [m]')
    plt.title('Average deviation of calculated distance to real distance\n as measured on 3000 Kitti image pairs')
    plt.show()
    #fig.savefig('data/plots/Stereo_Recognition_Rate_per_Distance.png')





if __name__ == "__main__":
    plot_xy_deviation_per_distance()