from image_processing.camera_model import IntrinsicModel
import numpy as np
import math
from image_processing.simulation import Car

# pixels on target according to the Johnson's Criteria optimistic an pessimistic pixels
# it is multiplied by 2 because it detects in line pairs and the minimum line pair in pixels equals to 2 pixels
johnsons_criteria = [
        {"detection":{"line_pairs":1,"pixels":1*2,"tolerance":0.25}},
        {"orientation":{"line_pairs":1.4,"pixels":1.4*2,"tolerance":0.35}},
        {"recognition":{"line_pairs":4,"pixels":4*2,"tolerance":0.8}},
        {"identification":{"line_pairs":6.4,"pixels":6.4*2,"tolerance":1.5}}
    ]

# assume for a car I need the below pixels/m
car_pixels_per_m = {
        "detection":1.6 * 3.28084, # conversion from feet to meters
        "recognition":2.7 * 3.28084,
        "identification":40 * 3.28084
    }

def isInFieldOfView(cameraFov,carAngle):
    lowCameraDetectionRange = 90-cameraFov/2
    highCameraDetectionRange = 90+cameraFov/2
    if (carAngle>=lowCameraDetectionRange and carAngle<=highCameraDetectionRange):
        return True
    else: return False

def getTargetVisibleSize(width,length,carRotation,carPositionAngle,carCoordinates):
    if (carPositionAngle==90):
        if (carRotation==0 or carRotation ==180): return length
        if (carRotation==90): return width
    #TODO double check the angle calculations
    if (carRotation>90): carRotation = 180 - carRotation
    carRotation = 90 - carRotation
    theta = carPositionAngle+carRotation
    targetVisibleSize = width*np.cos(theta)+length*np.sin(theta)
    return targetVisibleSize

def getDistance(target,image_pixels, visibleTargetArea, fov):
    return ((image_pixels / (target / visibleTargetArea)) * 360 / fov) / (2 * np.pi)

def estimateRange(goal="detection",
                  im = IntrinsicModel(focal_length=1,optical_center_x=0,optical_center_y=0,
                        ratio_image_coordinate_x=1,ratio_image_coordinate_y=1,pixel_skew=0,
                        fov_horizontal=np.pi/6,fov_vertical=np.pi/6,
                        pixel_size=1,image_width=640,image_height=480),
                  car=Car(x=20, y=15, length=4.7, width=1.9, theta=90)
                  ):
    car_length = car.length  # m
    car_width = car.width  # m
    car_rotation = car.theta # degrees from Y axis 0 = 180 = horizontal, 90 = vertical
    car_position = [car.x_coor,car.y_coor] #x, y

    camera_position = [0,0]

    carAngle = math.atan2(camera_position[1], camera_position[0]) - math.atan2(car_position[1], car_position[0]);
    carAngle = carAngle * 360 / (2 * np.pi);
    if (carAngle < 0):  carAngle = carAngle + 360
    if (isInFieldOfView(im.fov_horizontal,carAngle)):
        visibleTargetArea = getTargetVisibleSize(car_length,car_width,car_rotation,carAngle,car_position)
        return getDistance(car_pixels_per_m[goal],im.image_width,visibleTargetArea,im.fov_horizontal)
    else:
        return -1