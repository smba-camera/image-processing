import math

from image_processing.simulation import Weather
from image_processing.camera_model import *


def weatherEffectOnRange(
        weather = Weather(current_weather="Rain",size=1,density=150),
        estimatedRange = 200,
        camera_model = CameraModel(im=IntrinsicModel(focal_length=1, optical_center_x=0, optical_center_y=0, ratio_image_coordinate_x=1,
                 ratio_image_coordinate_y=1, pixel_skew=0, fov_horizontal=math.pi/6, fov_vertical=math.pi/6,
                 pixel_size=1, image_width=640, image_height=480), em=None, prepare_projection_matrix=True)):

    return -1