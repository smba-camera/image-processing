import numpy
import math
import numbers
import image_processing.util

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

    def __init__(self, rotation=None, translationVector=None):
        """Params: [rotation=rotationMatrix or rotationVector]"""
        if rotation:
            assert(len(rotation) == 3)
            if (isinstance(rotation[0], numbers.Number)):
                # rotation vector defines rotations
                self.alpha = rotation[0]
                self.beta = rotation[1]
                self.gamma = rotation[2]
            else:
                assert(len(rotation[0]) == 3)
                # rotation is defined by given matrix
                self.rotationMatrix = numpy.matrix(rotation)
                assert(self.rotationMatrix.shape[0] == 3 and self.rotationMatrix.shape[1] == 3)
        else:
            self.alpha = 0 # rotation of x axis
            self.beta = 0 # rotation of y axis
            self.gamma = 0 # rotation of z axis
        if translationVector:
            assert(len(translationVector) == 3)
            self.translation_x = translationVector[0]  # camera position in real world
            self.translation_y = translationVector[1]
            self.translation_z = translationVector[2]
        else:
            self.translation_x = 0  # camera position in real world
            self.translation_y = 0
            self.translation_z = 0

    def getRotationMatrix(self):

        if (hasattr(self, 'rotationMatrix')):
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

    def getTranslationVector(self):
        return numpy.matrix([self.translation_x, self.translation_y, self.translation_z]).transpose()

    def getMatrix(self):
        R = self.getRotationMatrix()
        t = self.getTranslationVector()
        return numpy.concatenate((R, t), axis=1)

# The Camera Model consists of one intrinsic and at least one extrinsic model
class CameraModel:
    def __init__(self, im=None, em=None, prepare_projection_matrix=True):
        self.prepare_projection_matrix = prepare_projection_matrix
        if (im):
            self.intrinsic_model = im
        else:
            self.intrinsic_model = IntrinsicModel()


        self.apply_new_extrinsic_models(em)

    # precalculate projection matrix for faster projection
    def calculate_projection_matrix(self):
        one_vect_short = numpy.matrix([1,1,1])
        zero_vect = numpy.matrix([0,0,0,0])

        intr_mat = numpy.concatenate((self.intrinsic_model.getMatrix(), one_vect_short.transpose()), axis=1)
        projection_mat = intr_mat
        for e_model in self.extrinsic_models:
            e_mat = numpy.concatenate((e_model.getMatrix(), zero_vect))
            projection_mat = numpy.matmul(projection_mat, e_mat)

        self.projection_matrix = projection_mat

    # will set the new extrinsic models and (if 'prepare_projection_matrix' is set) precalculate the projection
    def apply_new_extrinsic_models(self, ems):
        self.extrinsic_models = []
        if (ems):
            if (type(ems) is list):
                self.extrinsic_models = ems
            else:
                self.extrinsic_models.append(ems)
        else:
            self.extrinsic_models.append(ExtrinsicModel())
        if self.prepare_projection_matrix:
            self.calculate_projection_matrix()

    def projectToImage(self, coords):
        assert(len(coords) == 3)
        pos_vect = numpy.matrix(coords).transpose()
        one_vect = numpy.matrix([1])

        Ixyz = None
        if (hasattr(self, 'projection_matrix')):
            pos_vect = numpy.concatenate((pos_vect, one_vect))
            Ixyz = numpy.matmul(self.projection_matrix, pos_vect)
        else:
            for m in self.extrinsic_models:
                # append 1 at end because extrinsic matrices also contain translation
                pos_vect = numpy.concatenate((pos_vect, one_vect))
                pos_vect = numpy.matmul(m.getMatrix(), pos_vect)

            Ixyz = numpy.matmul(self.intrinsic_model.getMatrix(), pos_vect)

        xz = Ixyz[0][0]
        yz = Ixyz[1][0]
        z = Ixyz[2][0]
        if (z == 0): return [xz, yz]

        return  [
             float(xz / z),
             float(yz / z)
        ]

    def projectToWorld(self, coords):
        # TODO make it work with multiple extrinsic matrices
        assert(len(coords) == 2)
        result = numpy.matrix([coords[0], coords[1], 1]).transpose()
        # intrinsic back calculation
        intr_invert = numpy.linalg.inv(self.intrinsic_model.getMatrix())
        result = numpy.matmul(intr_invert, result)
        # extrinsic back calculation
        for m in self.extrinsic_models:
            rot_inverted = numpy.linalg.inv(m.getRotationMatrix())
            translation = m.getTranslationVector() # numpy.matmul(rot_inverted, )
            vector = numpy.matmul(rot_inverted, result)
            result = numpy.subtract(vector, translation)
        return image_processing.util.Vector3D(result, vector)
