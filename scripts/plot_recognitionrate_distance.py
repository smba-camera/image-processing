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
    drives=['0001','0005','0009','0015','0019','0022','0028','0056']

    startFrame = 0
    maxFrame = 100
    alpha=10
    stepsize=10
    maxrange=80
    values = [0]*((maxrange/stepsize)+1)
    hitmap = [0]*((maxrange/stepsize)+1)
    matcher = compare.VehicleDetectionAnalyization()
    fig = plt.figure()
    distances = np.linspace(1, maxrange, (maxrange/stepsize)+1,dtype=int)
    for drive in drives:
        datapath_left = drive+'_03_t200'
        datapath_right = drive+'_02_t200'
        matcher.runComparison(date,drive,datapath_left,datapath_right,alpha,stepsize)
    i=0
    for distance in distances:
        if not distance in matcher.error_rate_per_distance:
            values[i] = None
            i+=1
            continue
        values[i]=(matcher.error_rate_per_distance[distance])
        i+=1

    distances = [x-5 for x in distances]
    plt.plot(distances,values,'bo',distances,values,'k')
    plt.axis([0,maxrange,0,1])
    plt.ylabel('Stereo detection probability')
    plt.xlabel('Distance from detected cars [m]')
    plt.title('Stereo recognition rate per distance to detected cars\n as measured on 3000 Kitti image pairs')
    fig.savefig(os.path.join('data', 'plots', 'plot_recognitionrate_distance.png'))
    plt.show()
    #fig.savefig('data/plots/Stereo_Recognition_Rate_per_Distance.png')





if __name__ == "__main__":
    plot_recognitionrate_distance()