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

''' uses vehicle detection to mark all vehicles on the kitti images '''

def runStereoVisualization():
    path = os.path.abspath(os.path.join('data', 'kitti'))
    date = '2011_09_26'
    sync_foldername = "{}_drive_0001_sync".format(date)
    images_path = os.path.join(path, date, sync_foldername, date, sync_foldername)
    images_path_2 = os.path.join(images_path, "image_02","data")
    images_path_3 = os.path.join(images_path, "image_03","data")
    print(images_path_3)
    kittiDataLoader=kitti(path,date)
    leftCameraModel=kittiDataLoader.getCameraModel(3)
    rightCameraModel=kittiDataLoader.getCameraModel(2)
    sampleimg=cv2.imread(os.path.join(images_path_2,'0000000000.png'))
    detector=vd.VehicleDetection(sampleimg)
    images2=glob.glob(os.path.join(images_path_2,'*.png'))
    images2.sort()
    images3 = glob.glob(os.path.join(images_path_3, '*.png'))
    images3.sort()
    fig=plt.figure()
    positionEstimator= pe.PositionEstimationStereoVision(leftCameraModel,rightCameraModel)
    for image2,image3 in zip(images2,images3)[80:]:
        img2=cv2.imread(image2)
        img3=cv2.imread(image3)
        carsLeft = detector.find_vehicles(img3)
        carsRight= detector.find_vehicles(img2,False)
        matchedCars = vm.match_vehicles_stereo(carsLeft,carsRight)
        carPositions=[]
        for pair in matchedCars:
            if pair[0]!=None and pair[1]!=None:
                carPositions.append(positionEstimator.estimate_position_camera(pair[0], pair[1]))
        if not plt.get_fignums():
            # window has been closed
            return

        # print ('new Image:')
        ax1 = fig.add_subplot(211)
        plt.imshow(img3,cmap='gray')

        ax2 = fig.add_subplot(212)

        # TODO: use vehicle positions from image analysis
        count = len(carPositions)
        for j in range(count):
            ax2.add_patch(
                patches.Rectangle((-np.array(carPositions[j]).tolist()[0][0], -np.array(carPositions[j]).tolist()[0][2]), 2, 5))
        print j+1
        ax2.set_ylim([0, 100])
        ax2.set_xlim([-25, 25])
        ax2.set_aspect(1)
        plt.pause(1)
        fig.clear()

if __name__ == "__main__":
    runStereoVisualization()
