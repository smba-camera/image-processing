import os
import sys
# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

import math
import numpy
import matplotlib
from image_processing.kitti_data import Kitti,VehiclePositions
from image_processing.camera_model import ExtrinsicModel,CameraModel
from image_processing.position_estimation import PositionEstimationStereoVision
from image_processing.util import distance

import plot_projection


path = os.path.abspath(os.path.join('data', 'kitti'))
date = '2011_09_26'
drive_num = '0001'
camera_num_2 = 2
camera_num_3 = 3
image_frame = 42


def do_stuff():

    kitti = Kitti(path, date)
    vehiclePositions = VehiclePositions(path, date, drive_num)
    velo_extr_mat = kitti.getVeloExtrinsicModel()
    vehiclePositions_relative_to_camera_0 = []
    for vehicle in vehiclePositions.getVehiclePosition(image_frame):
        v_coords = [vehicle.xPos, vehicle.yPos, vehicle.zPos]
        new_coord = velo_extr_mat.project_coordinates(v_coords)
        vehiclePositions_relative_to_camera_0.append(new_coord)

    vehicles_on_image_2 = []
    vehicles_on_image_3 = []
    for v_coords in vehiclePositions_relative_to_camera_0:
        v_img_2_coords = kitti.getCameraModel(camera_num_2).projectToImage(v_coords)
        v_img_3_coords = kitti.getCameraModel(camera_num_3).projectToImage(v_coords)
        vehicles_on_image_2.append(v_img_2_coords)
        vehicles_on_image_3.append(v_img_3_coords)

    mean_distance_to_correct = []
    angles = [0, math.pi/16, math.pi/8,math.pi/4]
    for rad_turn in angles:
        # modify camera model 3 to simulate errors in calibration
        cm_03 = kitti.getCameraModel(camera_num_3)
        ems = cm_03.getExtrinsicModels()
        im = cm_03.getIntrinsicModel()

        # rotate extrinsic model 0 of camera 3
        new_extrinsic_model = rotateExtrtinsicModel(ems[0], rad_turn)
        new_ems = [new_extrinsic_model] + ems[1:]
        new_cm = CameraModel(im=cm_03.getIntrinsicModel(),em=new_ems)

        # calculate new estimated position
        cm_02 = kitti.getCameraModel(camera_num_2)
        position_estimator = PositionEstimationStereoVision(camera_model_one=cm_02, camera_model_two=new_cm)
        wrong_real_world_vehicles = []
        distances_to_correct = []
        #print(cm_02.getCameraPosition())
        for v_img_02,v_img_03,v_real_coord in zip(vehicles_on_image_2,vehicles_on_image_3,vehiclePositions_relative_to_camera_0)[1:]:
            new_wrong_pos = position_estimator.estimate_position(v_img_02,v_img_03)
            #unmodified_pos_estimation = PositionEstimationStereoVision(camera_model_one=cm_02, camera_model_two=cm_03).estimate_position(v_img_02,v_img_03,real_pos=v_real_coord, plot_func=plot_projection.plot_in_3d)

            wrong_real_world_vehicles.append(new_wrong_pos)
            #print("wrong: {}, correct: {}".format(new_wrong_pos, v_real_coord))
            distances_to_correct.append(distance(new_wrong_pos, v_real_coord))
        mean_distance_to_correct.append(sum(distances_to_correct) / len(distances_to_correct))

    plot(angles,mean_distance_to_correct)
    matplotlib.pyplot.show()




def rotateExtrtinsicModel(em, rad_turn):
    direction = em.getDirection()
    turn_em = ExtrinsicModel(rotation=[0,0,rad_turn])
    new_direction = turn_em.project_coordinates(direction)
    new_extrinsic_model = ExtrinsicModel(direction=new_direction, translationVector=em.getTranslation())
    return new_extrinsic_model

def plot(rads,distances):
    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.axes()
    ax.plot(rads,distances)


do_stuff()