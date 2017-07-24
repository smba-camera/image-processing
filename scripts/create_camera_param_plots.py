import os
import sys
# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

import math
from image_processing.kitti_data import Kitti,VehiclePositions
from image_processing.camera_model import ExtrinsicModel
import numpy

path = os.path.abspath(os.path.join('data', 'kitti'))
date = '2011_09_26'
drive_num = 1
camera_num_2 = 2
camera_num_3 = 3
image_frame = 42

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

for rad_turn in reversed([math.pi/4, math.pi/8, math.pi/16, 0]):
    # modify camera model 3 to simulate errors in calibration
    cm = kitti.getCameraModel(camera_num_3)
    ems = cm.getExtrinsicModels()
    im = cm.getIntrinsicModel()

    em = ems[0]
    direction = em.getDirection()
    turn_em = ExtrinsicModel(rotation=[rad_turn,0,0])
    new_direction = turn_em.project_coordinates(direction)
    new_extrinsic_model = ExtrinsicModel(direction=new_direction, translationVector=em.getTranslation())
    new_ems = [new_extrinsic_model] + ems[1:]



