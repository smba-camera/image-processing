import numpy
import math

class IntrinsicModel:
    # todo: load intrinsic parameters from config file
    # https://docs.python.org/3/library/configparser.html
    def __init__(self, focal_length=1, pixel_size=1, optical_center_x=0, optical_center_y=0, ratio_image_coordinate_x=1,
                 ratio_image_coordinate_y=1, pixel_skew=0):
        self.focal_length = focal_length
        self.pixel_size = pixel_size
        self.optical_center_x = optical_center_x # optical center on the image is on coordinates (0,0)
        self.optical_center_y = optical_center_y
        self.ratio_image_coordinate_x = ratio_image_coordinate_x # ratio between real world coordinates and image coordinates
        self.ratio_image_coordinate_y = ratio_image_coordinate_y
        self.pixel_skew = pixel_skew # pixels are not rectangular

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

class ExtrinsicModel:
    def __init__(self, rotationMatrix=None):
        if rotationMatrix:
            # rotation is defined by given matrix
            self.rotationMatrix = rotationMatrix
        else:
            self.alpha = 0 # rotation of x axis
            self.beta = 0 # rotation of y axis
            self.gamma = 0 # rotation of z axis

        self.translation_x = 0  # camera position in real world
        self.translation_y = 0
        self.translation_z = 0

    def getRotationMatrix(self):
        if (self.rotationMatrix):
            return self.rotationMatrix

        cosA = math.cos(self.alpha)
        sinA = math.sin(self.alpha)
        Rx = numpy.matrix([
            [1 , 0    , 0],
            [0 , cosA , - sinA],
            [0 , sinA , cosA]
        ])
        cosB = math.cos(self.beta)
        sinB = math.sin(self.beta)
        Ry = numpy.matrix([
            [cosB , 0 , sinB],
            [0    , 1 , 0],
            [-sinB, 0 , cosB],
        ])
        cosC = math.cos(self.gamma)
        sinC = math.sin(self.gamma)
        Rz = numpy.matrix([
            [cosC , -sinC , 0],
            [sinC , cosC , 0],
            [0    , 0    , 1]
        ])
        return numpy.matmul(Rx, numpy.matmul(Ry, Rz))

    def getMatrix(self):
        R = self.getRotationMatrix()
        t = numpy.matrix([self.translation_x, self.translation_y, self.translation_z]).transpose()
        return numpy.concatenate((R, t), axis=1)



class CameraModel:
    def __init__(self, im=None, em=None):
        if (im):
            self.intrinsic_model = im
        else:
            self.intrinsic_model = IntrinsicModel()
        if (em):
            self.extrinsic_model = em
        else:
            self.extrinsic_model = ExtrinsicModel()

    def projectToImage(self, coords):
        assert(len(coords) == 3)
        coordsMat = numpy.matrix(coords + [1]).transpose()
        Cxyz = numpy.matmul(self.extrinsic_model.getMatrix(), coordsMat)
        Ixyz = numpy.matmul(self.intrinsic_model.getMatrix(), Cxyz)
        xz = Ixyz[0][0]
        yz = Ixyz[1][0]
        z = Ixyz[2][0]
        if (z == 0): return [xz, yz]

        return  [
             xz / z,
             yz / z
        ]

