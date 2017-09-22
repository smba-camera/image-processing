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
    drives = ['0001','0005','0009','0015','0019','0022','0028','0056']
    thresholds=['200']

    startFrame = 0
    maxFrame = 15
    alpha=20
    stepsize=10
    maxrange=80
    fig = plt.figure()
    distances = np.arange(0, maxrange + 1, stepsize)
    values_per_threshold=[]

    matcher = compare.VehicleDetectionAnalyization()
    for drive in drives:
        datapath_left = "{}_03_t200".format(drive)
        datapath_right = "{}_02_t200".format(drive)
        matcher.runComparison(date,drive,datapath_left,datapath_right,alpha,stepsize)
    x_coords_real_found = []
    y_coords_real_found = []
    x_coords_found = []
    x_coords_notFound = []
    x_coords_falsePos = []
    y_coords_found = []
    y_coords_notFound = []
    y_coords_falsePos = []
    index_real = 0
    index_detected = 1
    for frame in matcher.matchedCars:
        for match in frame:

            if not match[index_real]:
                x_coords_falsePos.append(match[index_detected][0])
                y_coords_falsePos.append(match[index_detected][1])
                continue
            if not match[index_detected]:
                x_coords_notFound.append(match[index_real][0])
                y_coords_notFound.append(match[index_real][1])
                continue
            x_coords_real_found.append(match[index_real][0])
            y_coords_real_found.append(match[index_real][1])
            x_coords_found.append(match[index_detected][0])
            y_coords_found.append(match[index_detected][1])
    print (len(matcher.matchedCars))
    def negate(l):
        return [-x for x in l]
    # car:
    for i in range(len(x_coords_real_found)):
        plt.plot(negate([y_coords_found[i], y_coords_real_found[i]]), [x_coords_found[i], x_coords_real_found[i]], '0.8', label='Connection of detected\nwith real position' if not i else '')

    plt.plot(negate(y_coords_notFound), x_coords_notFound, 'r.', label='Non-detected cars', color=(1,0,0,0.3))
    plt.plot(negate(y_coords_found), x_coords_found, 'g.', label='Detected cars', color=(0,0.3,0,0.4), markersize=10)
    plt.plot(negate(y_coords_falsePos), x_coords_falsePos, 'y.', label='Wrong detections', markersize=5, color=(1,1,0,0.4))
    plt.plot([0,0], [-1,1], 'ks', label='Position of car',  markersize=13, color=(0,0,0))
    plt.axis('scaled')
    plt.axis([-40,40,-10,75])

    plt.xlabel('y-direction [m]')
    plt.ylabel('x-direction [m]')
    plt.title('Top View of (non-)detected cars in 3000 frames')
    #plt.axis([0,maxrange,0,1])
    handles, labels = plt.gca().get_legend_handles_labels()
    # reverse the order
    plt.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.04,1), loc="upper left", facecolor=(0.8,0.8,0.8,0.2))
    fig.savefig(os.path.join('data','plots','plot_vehicle_detection_distribution.png'))
    plt.show()

if __name__ == "__main__":
    plot_recognitionrate_distance()