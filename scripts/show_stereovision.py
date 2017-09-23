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
import image_processing.vehicle_detection.detect_vehicles_serialize as serialize
import image_processing.kitti_data.vehicle_positions as vp
import matplotlib.animation as manimation

''' uses vehicle detection to mark all vehicles on the kitti images '''

def runStereoVisualization():
    drive = '0056'
    thresh = '180'
    calculate=False
    path = os.path.abspath(os.path.join('data', 'kitti'))
    date = '2011_09_26'
    sync_foldername = "{}_drive_{}_sync".format(date,drive)
    images_path = os.path.join(path, date, sync_foldername, date, sync_foldername)
    images_path_2 = os.path.join(images_path, "image_02", "data")
    images_path_3 = os.path.join(images_path, "image_03", "data")
    print(images_path_3)
    kittiDataLoader = kitti(path, date)
    leftCameraModel = kittiDataLoader.getCameraModel(3)
    rightCameraModel = kittiDataLoader.getCameraModel(2)

    if not calculate:
        filename_r = serialize.get_detected_vehicles_file_name(drive,'02', thresh)
        filename_l = serialize.get_detected_vehicles_file_name(drive,'03', thresh)
        vehicles_l=serialize.load_detected_vehicles(filename_l)
        vehicles_r=serialize.load_detected_vehicles(filename_r)
    else:
        sampleimg=cv2.imread(os.path.join(images_path_2,'0000000000.png'))
        detector=vd.VehicleDetection(sampleimg)
    images2=glob.glob(os.path.join(images_path_2,'*.png'))
    images2.sort()
    images3 = glob.glob(os.path.join(images_path_3, '*.png'))
    images3.sort()
    fig=plt.figure()
    print images2
    positionEstimator= pe.PositionEstimationStereoVision(leftCameraModel,rightCameraModel)
    vehiclePositions = vp.VehiclePositions(path, date, drive)
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=7, metadata=metadata)
    video_path=os.path.join('data','videos')
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    file_name = os.path.join(video_path, "{}.mp4".format('stereovision'))
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()


 #   if frameid == 0:
 #       writer.saving(fig, file_name, 100)
 #   if frameid >= 0:
 #       writer.grab_frame()

    frameid=-1
    for image2,image3 in zip(images2,images3):
        frameid+=1
        img2=cv2.imread(image2)
        img3=cv2.imread(image3)
        if calculate:
            carsLeft = detector.find_vehicles(img3)
            carsRight= detector.find_vehicles(img2,False)
        else:
            carsLeft=vehicles_l[frameid]
            carsRight=vehicles_r[frameid]
        matchedCars = vm.match_vehicles_stereo(carsLeft,carsRight)
        carPositions=[]
        i=0
        for pair in matchedCars:
            if pair[0]!=None and pair[1]!=None:
                carPositions.append(positionEstimator.estimate_position(pair[0], pair[1]))
        vehicles=vehiclePositions.getVehiclePositions_projected(frameid)
       # plt.subplot(221)
        ax1=fig.add_subplot(221)
        # print ('new Image:')

        for i in range(len(carsLeft)):
            cv2.rectangle(img3, carsLeft[i][0], carsLeft[i][1],(0,0,255),5)
        ax1.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB),cmap='gray')

        ax2 = fig.add_subplot(222)
        for i in range(len(carsRight)):
            cv2.rectangle(img2, carsRight[i][0], carsRight[i][1], (0, 0, 255), 5)
        ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), cmap='gray')

        ax3 = fig.add_subplot(212)

        # TODO: use vehicle positions from image analysis
        count = len(carPositions)
        for j in range(count):
            ax3.add_patch(
                patches.Rectangle((-carPositions[j][0], -carPositions[j][2]), 2, 5))
        for j in range(len(vehicles)):
            ax3.add_patch(
                patches.Rectangle((vehicles[j][0],vehicles[j][2]), 2, 5, color='red'))


        ax3.set_ylim([0, 100])
        ax3.set_xlim([-25, 25])
        ax3.set_aspect(1)
        plt.pause(0.001)
        fig.clear()


if __name__ == "__main__":
    runStereoVisualization()
