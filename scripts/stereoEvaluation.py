import os
import glob,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

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
    path2 = os.path.abspath(os.path.join('data', 'images2'))
    path3 =os.path.abspath(os.path.join('data','images3'))
    sampleimg=cv2.imread(os.path.join(path2,'0000000000.png'))
    detector=vd.VehicleDetection(sampleimg)
    images2=glob.glob(os.path.join(path2,'*.png'))
    images2.sort()
    images3 = glob.glob(os.path.join(path3, '*.png'))
    images3.sort()
    fig=plt.figure()
    positionEstimator= pe.PositionEstimationStereoVision(leftCameraModel,rightCameraModel)
    framecount=80
    for image2,image3 in zip(images2[80:],images3[80:]):
        img2=cv2.imread(image2)
        img3=cv2.imread(image3)
        carsLeft = detector.find_vehicles(img3)
        carsRight= detector.find_vehicles(img2,False)
        matchedCars = vm.match_vehicles_stereo(carsLeft,carsRight)
        carPositions=[]
        for pair in matchedCars:
            if pair[0]!=None and pair[1]!=None:
                carPositions.append(positionEstimator.estimate3DPosition(pair[0],pair[1]))
        vehicles=vehiclePositions.getVehiclePosition(framecount)
        framecount+=1
        print carPositions
        print len(vehicles)
        for x in range(len(vehicles)):
            if vehicles[x].type=='Car':
                print vehicles[x].xPos,vehicles[x].yPos,vehicles[x].zPos


if __name__ == "__main__":
    runStereoEvaluation()