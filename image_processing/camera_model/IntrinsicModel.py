import numpy
import math

class IntrinsicModel:
    # todo: load intrinsic parameters from config file
    # https://docs.python.org/3/library/configparser.html
    def __init__(self, focal_length=1, optical_center_x=0, optical_center_y=0, ratio_image_coordinate_x=1,
                 ratio_image_coordinate_y=1, pixel_skew=0, fov_horizontal=math.pi/6, fov_vertical=math.pi/6,
                 pixel_size=1, image_width=100, image_height=100):

        self.focal_length = focal_length
        self.optical_center_x = optical_center_x # optical center on the image is on coordinates (0,0)
        self.optical_center_y = optical_center_y
        self.ratio_image_coordinate_x = ratio_image_coordinate_x # ratio between real world coordinates and image coordinates
        self.ratio_image_coordinate_y = ratio_image_coordinate_y
        self.pixel_skew = pixel_skew # pixels are not rectangular
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical

        self.pixel_size = pixel_size
        self.image_width = image_width
        self.image_height = image_height

    def image_pixel_width(self):
        return self.image_width / self.pixel_size

    def image_pixel_height(self):
        return self.image_height / self.pixel_size

    def getMatrix(self):
        alpha = self.focal_length * self.ratio_image_coordinate_x
        beta = self.focal_length * self.ratio_image_coordinate_y
        s = self.pixel_skew
        u0 = self.optical_center_x
        v0 = self.optical_center_y

        return numpy.matrix([
            [alpha , s    , u0],
            [0     , beta , v0],
            [0     , 0    , 1]
        ])
