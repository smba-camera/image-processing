import os
import glob,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

import image_processing.vehicle_detection.detect_vehicles_serialize as serialize
import image_processing.vehicle_detection.Vehicle_detection as vd
import image_processing.vehicle_detection.stereo_vision_vehicle_matcher as vm
import image_processing.position_estimation.position_estimation as pe
import image_processing.kitti_data.Kitti as kitti
import image_processing.kitti_data.vehicle_positions as vp

''' uses vehicle detection to mark all vehicles on the kitti images '''

def runStereoEvaluation():
    path = os.path.abspath(os.path.join('data', 'kitti'))
    date = '2011_09_26'
    kittiDataLoader=kitti(path,date)
    vehiclePositions=vp.VehiclePositions(path,date,56)
    leftCameraModel=kittiDataLoader.getCameraModel(3)
    rightCameraModel=kittiDataLoader.getCameraModel(2)
    fig=plt.figure()
    positionEstimator= pe.PositionEstimationStereoVision(leftCameraModel,rightCameraModel)
    datapath_left='0056_03_0-10_t975'
    datapath_right = '0056_02_0-10_t975'
    startFrame=0
    maxFrame=10
    detectedCarCount=0
    realCarCount=0
    ax1 = fig.add_subplot(211)
    carsLeft = serialize.load_detected_vehicles(datapath_left)
    carsRight = serialize.load_detected_vehicles(datapath_right)
    for framecount in range(startFrame,maxFrame):
        matchedCars = vm.match_vehicles_stereo(carsLeft[framecount],carsRight[framecount])
        carPositions=[]
        for pair in matchedCars:
            if pair[0]!=None and pair[1]!=None:
                list=positionEstimator.estimate_position(pair[0], pair[1])
                temp=[-list[2],list[0]]
                carPositions.append(temp)

        #carPositions.sort(key=lambda tup:np.sqrt(tup[0]*tup[0]+tup[1]*tup[1])
        vehicles=vehiclePositions.getVehiclePosition(framecount)
        cars=[]
        for x in range(len(vehicles)):
            if vehicles[x].type=='Car':
                cars.append((vehicles[x].xPos,vehicles[x].yPos))
        realCarCount+=len(cars)
        detectedCarCount+=len(carPositions)
        #cars.sort(key=lambda tup: np.sqrt(tup[0] * tup[0] + tup[1] * tup[1]))


        for i in range(len(carPositions)):
            ax1.add_patch(patches.Circle((cars[i][1], cars[i][0]), 0.05, color='Green'))
            print cars[i]
        for j in range(i+1,len(cars)):
            ax1.add_patch(patches.Circle((cars[j][1], cars[j][0]), 0.05, color='Red'))
            print cars[j]

    ax1.set_ylim([5, 50])
    ax1.set_xlim([-15, 15])
    ax1.set_aspect(1)
    plt.pause(10)


if __name__ == "__main__":
    runStereoEvaluation()